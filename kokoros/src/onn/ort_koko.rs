use std::borrow::Cow;

use super::ort_base;
use crate::utils::debug::format_debug_prefix;
use ndarray::{ArrayBase, IxDyn, OwnedRepr};
use ort::{
    session::{Session, SessionInputValue, SessionInputs},
    value::{Tensor, Value},
};
use ort_base::OrtBase;

pub const STYLE: &str = "style";
pub const SPEED: &str = "speed";

/// Resolved ONNX tensor names for a loaded model.
///
/// Kokoro-family models are exported with slightly different input/output
/// names depending on the origin (e.g. the standard hexgrad export vs. the
/// Kikiri/StyleTTS2-based German Martin export). Instead of guessing the schema
/// from the output count we resolve the real names from the loaded session so
/// any compatible model runs without changes.
#[derive(Debug, Clone)]
pub struct ModelSchema {
    /// Token input tensor name (e.g. "tokens" or "input_ids").
    pub tokens: String,
    /// Audio output tensor name (e.g. "waveform" or "audio").
    pub audio: String,
    /// Optional per-frame duration output (e.g. "duration" or "durations").
    pub duration: Option<String>,
}

impl ModelSchema {
    /// Finds the first session input whose name matches one of the candidates.
    fn resolve_input(sess: &Session, candidates: &[&str]) -> Option<String> {
        sess.inputs
            .iter()
            .map(|input| input.name.as_str())
            .find(|name| candidates.contains(name))
            .map(str::to_owned)
    }

    /// Finds the first session output whose name matches one of the candidates.
    fn resolve_output(sess: &Session, candidates: &[&str]) -> Option<String> {
        sess.outputs
            .iter()
            .map(|output| output.name.as_str())
            .find(|name| candidates.contains(name))
            .map(str::to_owned)
    }

    /// Builds the schema from the actual inputs/outputs of a loaded session.
    fn from_session(sess: &Session) -> Self {
        let tokens = Self::resolve_input(sess, &["tokens", "input_ids"])
            .unwrap_or_else(|| "tokens".to_owned());
        let audio = Self::resolve_output(sess, &["waveform", "audio", "waveforms"])
            .unwrap_or_else(|| "audio".to_owned());
        let duration = Self::resolve_output(sess, &["duration", "durations"]);

        tracing::info!(
            "OrtKoko: resolved schema tokens={} audio={} duration={:?}",
            tokens,
            audio,
            duration
        );

        Self {
            tokens,
            audio,
            duration,
        }
    }
}

/// Distinguishes whether the model emits per-frame duration/timestamps.
pub enum ModelStrategy {
    Standard(Session),
    Timestamped(Session),
}

pub struct OrtKoko {
    inner: Option<ModelStrategy>,
    schema: ModelSchema,
}

impl ModelStrategy {
    fn is_timestamped(&self) -> bool {
        matches!(self, ModelStrategy::Timestamped(_))
    }

    fn sess(&self) -> &Session {
        match self {
            ModelStrategy::Standard(sess) => sess,
            ModelStrategy::Timestamped(sess) => sess,
        }
    }

    fn sess_mut(&mut self) -> &mut Session {
        match self {
            ModelStrategy::Standard(sess) => sess,
            ModelStrategy::Timestamped(sess) => sess,
        }
    }
}

impl OrtBase for OrtKoko {
    fn set_sess(&mut self, sess: Session) {
        let schema = ModelSchema::from_session(&sess);

        let strategy = if schema.duration.is_some() {
            tracing::info!("OrtKoko: Timestamped backend activated (duration output present)");
            ModelStrategy::Timestamped(sess)
        } else {
            tracing::info!("OrtKoko: Standard backend activated (no duration output)");
            ModelStrategy::Standard(sess)
        };

        self.inner = Some(strategy);
        self.schema = schema;
    }

    fn sess(&self) -> Option<&Session> {
        self.inner.as_ref().map(ModelStrategy::sess)
    }
}

impl OrtKoko {
    pub fn new(model_path: String) -> Result<Self, String> {
        let mut instance = OrtKoko {
            inner: None,
            // Placeholder, overwritten in `set_sess` via `load_model`.
            schema: ModelSchema {
                tokens: "tokens".to_owned(),
                audio: "audio".to_owned(),
                duration: None,
            },
        };
        instance.load_model(model_path)?;
        Ok(instance)
    }

    pub fn strategy(&self) -> Option<&ModelStrategy> {
        self.inner.as_ref()
    }

    /// Whether the loaded model provides per-frame duration data (timestamps).
    pub fn supports_timestamps(&self) -> bool {
        matches!(self.inner.as_ref(), Some(ModelStrategy::Timestamped(_)))
    }

    fn prepare_inputs(
        tokens_key: &str,
        tokens: Vec<Vec<i64>>,
        styles: Vec<Vec<f32>>,
        speed: f32,
    ) -> Result<Vec<(Cow<'static, str>, SessionInputValue<'static>)>, Box<dyn std::error::Error>>
    {
        let shape = [tokens.len(), tokens[0].len()];
        let tokens_tensor =
            Tensor::from_array((shape, tokens.into_iter().flatten().collect::<Vec<i64>>()))?;

        let shape_style = [styles.len(), styles[0].len()];
        let style_tensor = Tensor::from_array((
            shape_style,
            styles.into_iter().flatten().collect::<Vec<f32>>(),
        ))?;

        let speed_tensor = Tensor::from_array(([1], vec![speed]))?;

        Ok(vec![
            (
                Cow::Owned(tokens_key.to_owned()),
                SessionInputValue::Owned(Value::from(tokens_tensor)),
            ),
            (
                Cow::Borrowed(STYLE),
                SessionInputValue::Owned(Value::from(style_tensor)),
            ),
            (
                Cow::Borrowed(SPEED),
                SessionInputValue::Owned(Value::from(speed_tensor)),
            ),
        ])
    }

    pub fn infer(
        &mut self,
        tokens: Vec<Vec<i64>>,
        styles: Vec<Vec<f32>>,
        speed: f32,
        request_id: Option<&str>,
        instance_id: Option<&str>,
        chunk_number: Option<usize>,
    ) -> Result<(ArrayBase<OwnedRepr<f32>, IxDyn>, Option<Vec<f32>>), Box<dyn std::error::Error>>
    {
        let debug_prefix = format_debug_prefix(request_id, instance_id);
        let chunk_info = chunk_number
            .map(|n| format!("Chunk: {}, ", n))
            .unwrap_or_default();
        tracing::debug!(
            "{} {}inference start. Tokens: {}",
            debug_prefix,
            chunk_info,
            tokens.len()
        );

        let strategy = self.inner.as_mut().ok_or("Session is not initialized.")?;
        let inputs = Self::prepare_inputs(&self.schema.tokens, tokens.clone(), styles, speed)?;
        let outputs = strategy.sess_mut().run(SessionInputs::from(inputs))?;

        let audio_key = self.schema.audio.as_str();
        let (shape, data) = outputs[audio_key]
            .try_extract_tensor::<f32>()
            .or_else(|_| outputs["waveform"].try_extract_tensor::<f32>())
            .or_else(|_| outputs["audio"].try_extract_tensor::<f32>())
            .map_err(|_| format!("Model: Could not find a valid audio output ('{}')", audio_key))?;

        let shape_vec: Vec<usize> = shape.into_iter().map(|&i| i as usize).collect();
        let audio_array = ArrayBase::from_shape_vec(shape_vec, data.to_vec())?;

        let duration = match &self.schema.duration {
            Some(key) => {
                let value = &outputs[key.as_str()];
                let extracted = value
                    .try_extract_tensor::<f32>()
                    .map(|(_, d)| d.to_vec())
                    .or_else(|_| {
                        value
                            .try_extract_tensor::<i64>()
                            .map(|(_, d)| d.iter().map(|&v| v as f32).collect::<Vec<f32>>())
                    })
                    .map_err(|_| format!(
                        "Timestamped Model Error: Expected output tensor '{}' of type f32 or i64.",
                        key
                    ))?;
                Some(extracted)
            }
            None => None,
        };

        Ok((audio_array, duration))
    }
}
