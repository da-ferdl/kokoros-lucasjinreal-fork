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

.venv/bin/python ../../scripts/male_philip_emotions/make_voices_philip.py \
    --checkpoint stage1_ep4_raw.pth \
    --wav-dir philip_source_emotion_wavs \
    --output voices-philip.npz
```

Inputs (downloaded from `kikiri-tts/kikiri-german-base-51speakers-synthetic`):

- `stage1_ep4_raw.pth` — StyleTTS2 **Stage-1 raw** checkpoint. Contains the
  `net.style_encoder` / `net.predictor_encoder` needed to compute voice styles.
  The converted `kikiri_german_base_51spk_ep4.pth` only holds Kokoro-inference
  weights (no encoders) and is **not** used here.
- `philip_source_emotion_wavs/*.wav` — one source WAV per emotion, named by the
  emotion (the stem becomes the npz key). Mono, any sample rate (resampled to
  24 kHz internally).

## How it works

The German Kokoro-82M ONNX decoder has no reference encoder (its only inputs
are `tokens`/`style`/`speed`); voices are pre-trained StyleTTS2 embeddings of
shape `(510, 1, 256)` (128 acoustic/timbre dims + 128 prosodic dims). The
script reproduces the upstream `kikiri-tts` `extract_voicepack.py` logic: for
each emotion WAV it computes a mel spectrogram (24 kHz, n_fft=2048,
win=1200, hop=300, 80 mels, norm mean=-4/std=4), runs `style_encoder` +
`predictor_encoder`, averages the vectors, and writes one `(510, 1, 256)`
pack per emotion into the npz (all 510 rows identical per emotion).

Because only a Stage-1 checkpoint is supplied, the `predictor_encoder` is
untrained (documented StyleTTS2 behaviour); the script uses `style_encoder`
for both halves exactly as upstream does.

At runtime the Rust `TTSKoko` `load_voices`/`mix_styles` reads these npz keys
unchanged — no Rust code changes are needed to use the new emotion voices.

## Run the example

From the repo root:

```bash
cargo run --release --example german_philip_emotions
```

This synthesizes the test sentence in all eight emotions and plays them
sequentially.
