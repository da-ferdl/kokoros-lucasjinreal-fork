//! This example is here to add different emotion variations.
//!
//! The goal should be to use eg, instead of a style-name "martin", style-name "happy" - one of the philip-voice variations.
//!
//! The directory with the source files and kokoro model is `<repo-root-dir>/_dev_data/male_philip_emotion_variation`.
//! The source philip-wav files are in the directory `<repo-root-dir>/_dev_data/male_philip_emotion_variation/philip_source_emotion_wavs`.
//!
//! Run within the repo root directory:
//! `cargo run --release --example german_philip_emotions`
//!

use anyhow::Result;
use kokoros::tts::koko::TTSKoko;
use rodio::{DeviceSinkBuilder, Player, Source};
use std::{num::NonZero, path::PathBuf, time::Duration};

fn main() -> Result<()> {
    tracing_subscriber::fmt()
        .with_env_filter(
            tracing_subscriber::EnvFilter::try_from_default_env()
                .unwrap_or_else(|_| tracing_subscriber::EnvFilter::new("info")),
        )
        .init();

    let test_text = "Hallo, ich bin Robi und presentiere euch die TTS Engine Kokoro.";

    let base_dir = PathBuf::from("_dev_data/male_philip_emotion_variation");

    let model_path = base_dir.join("kokoro-martin.onnx");
    let voices_path = base_dir.join("voices-philip.npz");

    let rt = tokio::runtime::Builder::new_multi_thread()
        .enable_time()
        .enable_io()
        .worker_threads(2)
        .thread_name("tts-load-worker")
        .build()
        .unwrap();

    let model = rt.block_on(async move {
        TTSKoko::new(model_path.to_str().unwrap(), voices_path.to_str().unwrap()).await
    });

    let mut sink = DeviceSinkBuilder::open_default_sink()?;
    sink.log_on_drop(false);
    let player = Player::connect_new(&sink.mixer());

    /*** Following does currently not work of course - but that is the goal  ***/

    let samples_neutral = model
        .tts_raw_audio(test_text, "de", "neutral", 1.0, None, None, None, None)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    player.append(AudioSource::new(samples_neutral));

    let samples_happy = model
        .tts_raw_audio(test_text, "de", "happy", 1.0, None, None, None, None)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    player.append(AudioSource::new(samples_happy));

    let samples_sad = model
        .tts_raw_audio(test_text, "de", "sad", 1.0, None, None, None, None)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    player.append(AudioSource::new(samples_sad));

    let samples_angry = model
        .tts_raw_audio(test_text, "de", "angry", 1.0, None, None, None, None)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    player.append(AudioSource::new(samples_angry));

    let samples_fearful = model
        .tts_raw_audio(test_text, "de", "fearful", 1.0, None, None, None, None)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    player.append(AudioSource::new(samples_fearful));

    let samples_surprised = model
        .tts_raw_audio(test_text, "de", "surprised", 1.0, None, None, None, None)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    player.append(AudioSource::new(samples_surprised));

    let samples_disgusted = model
        .tts_raw_audio(test_text, "de", "disgusted", 1.0, None, None, None, None)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    player.append(AudioSource::new(samples_disgusted));

    let samples_bored = model
        .tts_raw_audio(test_text, "de", "bored", 1.0, None, None, None, None)
        .map_err(|e| anyhow::Error::msg(e.to_string()))?;
    player.append(AudioSource::new(samples_bored));

    // keeps the thread alive till audio ends.
    player.sleep_until_end();

    Ok(())
}

pub struct AudioSource {
    samples: Vec<f32>,
    next_index: usize,
    sample_rate: NonZero<u32>,
    channels: NonZero<u16>,
}
impl AudioSource {
    pub fn new(samples: Vec<f32>) -> Self {
        Self {
            samples,
            next_index: 0,
            // Safe - hardcoded non-zero value.
            sample_rate: unsafe { NonZero::new(24000).unwrap_unchecked() },
            // Safe - hardcoded non-zero value.
            channels: unsafe { NonZero::new(1_u16).unwrap_unchecked() },
        }
    }
}
impl Source for AudioSource {
    fn channels(&self) -> NonZero<u16> {
        self.channels
    }

    fn sample_rate(&self) -> NonZero<u32> {
        self.sample_rate
    }

    fn total_duration(&self) -> Option<Duration> {
        None
    }

    fn current_span_len(&self) -> Option<usize> {
        None
    }
}
impl Iterator for AudioSource {
    type Item = f32;

    fn next(&mut self) -> Option<Self::Item> {
        let i = self.next_index;

        if i == self.samples.len() {
            return None;
        }

        self.next_index += 1;

        Some(self.samples[i])
    }
}
