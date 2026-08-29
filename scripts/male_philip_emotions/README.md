# Philip voice — emotion variations

Generates `voices-philip.npz` with one voice key per emotion (`neutral`,
`happy`, `sad`, `angry`, `fearful`, `surprised`, `disgusted`, `bored`) from
the source emotion WAVs in `philip_source_emotion_wavs/`.

## Reproduce

Source files live un-tracked in `_dev_data/male_philip_emotion_variation/`
(large, gitignored). Run from there:

```bash
cd _dev_data/male_philip_emotion_variation

# Python 3.11 is required (torch has no wheels for 3.14 / this x86_64 mac).
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python 'numpy<2' torch torchaudio soundfile

# Recommended: style_encoder (timbre) from Stage-1, trained predictor_encoder
# (emotional prosody) from the target voice's Stage-2 fine-tune.
.venv/bin/python ../../scripts/male_philip_emotions/make_voices_philip.py \
    --style-encoder-model stage1_ep4_raw.pth \
    --predictor-encoder-model stage2_martin_ep10_raw.pth \
    --wav-dir philip_source_emotion_wavs \
    --output voices-philip.npz
```

Inputs:

- `stage1_ep4_raw.pth` — StyleTTS2 **Stage-1 raw** checkpoint; supplies the
  `style_encoder` (timbre). From `kikiri-tts/kikiri-german-base-51speakers-synthetic`.
- `stage2_martin_ep10_raw.pth` — **Stage-2 raw** fine-tune of the target voice;
  supplies the **trained `predictor_encoder`** (emotional prosody). From
  `kikiri-tts/kikiri-german-martin`. This is the checkpoint the
  `kokoro-martin.onnx` decoder was exported from.
- `philip_source_emotion_wavs/*.wav` — one source WAV per emotion, named by the
  emotion (the stem becomes the npz key). Mono, any sample rate (resampled to
  24 kHz internally).

The converted `kikiri_german_base_51spk_ep4.pth` / `kikiri_german_martin_ep10.pth`
only hold Kokoro-inference weights (no encoders) and are **not** usable here.

## How it works

The German Kokoro-82M ONNX decoder has no reference encoder (its only inputs
are `tokens`/`style`/`speed`); voices are pre-trained StyleTTS2 embeddings of
shape `(510, 1, 256)` (128 acoustic/timbre dims + 128 prosodic dims). The
script reproduces the upstream `kikiri-tts` `extract_voicepack.py` logic: for
each emotion WAV it computes a mel spectrogram (24 kHz, n_fft=2048,
win=1200, hop=300, 80 mels, norm mean=-4/std=4), runs `style_encoder` +
`predictor_encoder`, averages the vectors, and writes one `(510, 1, 256)`
pack per emotion into the npz (all 510 rows identical per emotion).

Because the emotion source WAVs share one voice (same timbre), the acoustic
half stays similar across emotions; the emotional differences are carried by
the **prosody half** (`predictor_encoder`). The Stage-1 `predictor_encoder` is
untrained/collapsed, which flattens emotion (all styles end up near-identical).
Using the Stage-2 `predictor_encoder` keeps the source WAVs' emotional prosody,
so the emotion variations remain audibly distinct. If only a Stage-1 checkpoint
is supplied, the script falls back to `style_encoder` for the prosody half
(muted emotion).

At runtime the Rust `TTSKoko` `load_voices`/`mix_styles` reads these npz keys
unchanged — no Rust code changes are needed to use the new emotion voices.

## Run the example

From the repo root:

```bash
cargo run --release --example german_philip_emotions
```

This synthesizes the test sentence in all eight emotions and plays them
sequentially.
