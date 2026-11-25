# Parrot-TTS

Vietnamese text-to-speech, made for Vietnamese speakers. It runs on your own computer: free, no account, and your text never leaves your machine.

Parrot-TTS is [Fun-CosyVoice3-0.5B](https://huggingface.co/FunAudioLLM/Fun-CosyVoice3-0.5B-2512) with its text-reading stage (`llm.pt`) retrained for Vietnamese, plus a Vietnamese text normalizer (numbers, dates, abbreviations) and a ready-made voice pack.

- Model and voices: [huggingface.co/rin98s/Parrot-TTS](https://huggingface.co/rin98s/Parrot-TTS)
- Code: this repository. The Python package and command are still called `vietvoice`.

## Requirements

- Python 3.12 and `git`.
- An NVIDIA GPU makes it fast. CPU-only machines (Macs included) work but are slow: an M-series Mac needs about 4–5 seconds per second of audio, plus about 30 seconds to load the model on each run.
- About 5 GB of disk space and bandwidth for the first install (model ~4.4 GB, libraries ~1 GB).

## Install

```bash
git clone https://github.com/RinNguyens/Parrot-TTS.git
cd Parrot-TTS
python3.12 -m venv .venv-tts && source .venv-tts/bin/activate
pip install -e ".[tts]"
vietvoice download
vietvoice check
```

`vietvoice download` fetches the CosyVoice code, the base model, the Parrot-TTS `llm.pt` and the voice pack into `~/.cache/vietvoice` (set `VIETVOICE_HOME` to use another folder). `vietvoice check` reports anything that is missing.

## Usage

```bash
vietvoice voices
vietvoice tts "Xin chào các bạn, hôm nay giảm 50% toàn bộ cửa hàng." --voice "Thanh Thanh" -o hello.wav
cat article.txt | vietvoice tts - --voice "Thành" -o article.wav
```

In Python:

```python
from vietvoice.tts import VietVoice, save

tts = VietVoice()
audio, sr = tts.speak("Xin chào các bạn.", voice="Thanh Thanh")
save("hello.wav", audio, sr)
```

Voice names work with or without Vietnamese diacritics (`"Thanh Thanh"`, `thanh_thanh`).

`vietvoice serve` starts a local HTTP reading server.

## Voices

The voice pack downloaded by `vietvoice download` contains:

| Voice | ID | Accent | Gender | Best for |
| --- | --- | --- | --- | --- |
| Thanh Thanh | `thanh_thanh` | Southern | Female | Podcasts, heart-to-heart talks, storytelling |
| Thành | `thanh` | Southern | Male | Radio reading, late-night stories |

Both original speakers agreed to have their voices published under these names.

Quality, measured by transcribing the output with Whisper on 167 held-out sentences and 38 tone minimal pairs:

| Voice | Character error rate (CER) | Tone errors | Similarity to reference voice |
| --- | --- | --- | --- |
| Thanh Thanh | 0.059 | 0.126 | 0.906 |
| Thành | 0.060 | 0.112 | 0.941 |

More voices, including Northern accents, are on the way.

### Use your own voice

Record one 5–15 second sentence in a quiet room and write down exactly what you said:

```bash
vietvoice tts "Văn bản cần đọc." --prompt-wav sample.wav --prompt-text "Exact words spoken in sample.wav" -o out.wav
```

Only use your own voice, or the voice of someone who has agreed to it. Do not use Parrot-TTS to imitate other people.

## AI-generated label

WAV files written by `vietvoice` carry a note in their metadata saying the audio is an AI-generated synthetic voice. When you publish audio, say clearly that the voice is synthetic.

## License

- **Source code** in this repository: Apache-2.0 (`LICENSE`).
- **Model and voice pack**: free to download and use for personal, educational, research and other non-commercial purposes. The current `llm.pt` was trained partly on data licensed for research and education only (PhoAudiobook). A commercially usable version will follow once the model is retrained on cleanly licensed data.
- Base model Fun-CosyVoice3-0.5B and the CosyVoice code: Apache-2.0, by FunAudioLLM.

Not allowed: commercial use of the model; cloning someone's voice without their consent; fraud, impersonation or disinformation.

Every library, model and dataset used, with its license: `LICENSES.md`.
