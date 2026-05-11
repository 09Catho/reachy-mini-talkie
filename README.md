---
title: The 1930 Broadcaster
emoji: 📻
colorFrom: yellow
colorTo: red
sdk: static
pinned: false
short_description: Reachy Mini speaks like a 1930 BBC wireless broadcaster
tags:
  - reachy_mini
  - reachy_mini_python_app
---

# 📻 The 1930 Broadcaster

> *"Good evening. This is the British Broadcasting Corporation."*

**Reachy Mini × Talkie-1930** — a one-click app that turns your Reachy Mini robot into a 1920s BBC wireless announcer. It doesn't just *sound* 1930. It *thinks* 1930.

---

## How it works

```
You speak  →  faster-whisper STT  →  Talkie-1930 LM  →  Kokoro TTS  →  Reachy speaks
                  (on-device)          (1930 prose)      (bm_george)    + head choreography
```

| Component | What it does | Runs where |
|---|---|---|
| [faster-whisper](https://github.com/SYSTRAN/faster-whisper) tiny.en | Transcribes your voice | On Reachy (Pi 5 CPU) |
| [Talkie-1930-13b-it](https://huggingface.co/talkie-lm/talkie-1930-13b-it) | Generates period-authentic 1930 English prose | Cloud (HF Space) |
| [Kokoro-82M](https://huggingface.co/hexgrad/Kokoro-82M) `bm_george` | Speaks in a measured Received Pronunciation baritone | On Reachy (Pi 5 CPU) |
| Choreography | Head bobs + antenna twitches synced to audio energy | On Reachy |

**The defining insight:** Talkie-1930 is a *language model*, not a voice model. It was trained on 260 billion tokens of pre-1931 text — etiquette manuals, newspapers, patent filings, parliamentary Hansard — and replies in the vocabulary and worldview of that era. Pair it with a British TTS and you have a robot that genuinely *inhabits* 1930.

Ask it about electricity. Ask it about the wireless. Ask it about Twitter. Different vibes.

---

## Installation

### Requirements

- Reachy Mini Wireless (Raspberry Pi 5, 8 GB RAM)
- Network access (Talkie-1930 runs on a Hugging Face Space)
- `espeak-ng` for Kokoro phonemisation: `sudo apt-get install espeak-ng`
- Python ≥ 3.10

### From source

```bash
git clone https://github.com/09Catho/reachy-mini-talkie
cd reachy-mini-talkie
sudo apt-get install espeak-ng
uv pip install -e "."
python -m reachy_mini_talkie         # smoke-test with mock robot
```

### With MuJoCo simulator (no hardware needed)

```bash
uv pip install -e ".[sim]"
python -m reachy_mini_talkie
```

---

## Usage

Once started, the robot enters **LISTENING** mode (head tilts forward, antennae neutral). Speak naturally. It will:

1. **THINKING** — head tilts, slow side-to-side sway while querying Talkie
2. **SPEAKING** — replies in 1930 prose, head bobs to audio energy, antennae twitch on punctuation

Conversation history is maintained across turns (rolling 6-turn window).

### Push-to-talk mode

For noisy demo environments, start the app with `push_to_talk=True`:

```python
from reachy_mini_talkie import TalkieApp
app = TalkieApp(push_to_talk=True)
```

Then call `app._vad.start_speaking()` / `.stop_speaking()` from a button handler.

### Changing the voice

```python
app = TalkieApp(voice="bf_emma")   # female BBC-announcer
app = TalkieApp(voice="bm_fable")  # theatrical male
```

Available British voices (Kokoro VOICES.md): `bm_george` *(default)*, `bm_fable`, `bf_emma`, `bf_isabella`.

---

## Development & testing

```bash
# Unit test each component independently
python -m pytest tests/ -v

# Full pipeline smoke-test (no hardware needed)
python -m reachy_mini_talkie

# With MuJoCo simulator
python -c "
from reachy_mini import ReachyMini
import threading, reachy_mini_talkie
stop = threading.Event()
with ReachyMini(simulation=True) as r:
    reachy_mini_talkie.TalkieApp().run(r, stop)
"
```

### Latency budget

| Stage | Target |
|---|---|
| STT (faster-whisper tiny.en, Pi 5 CPU) | ~1.0 s |
| Talkie LM round-trip (HF Space, ~80 tokens) | ~3–4 s |
| TTS first chunk (Kokoro, Pi 5 CPU) | ~0.6 s |
| **Mouth-to-ear total** | **< 6 s** |

The "thinking" choreography bridges perceived latency so the robot is never silently frozen.

---

## Risks & known limitations

| Issue | Mitigation |
|---|---|
| Talkie Space rate-limits or sleeps | Retry with backoff; polite fallback line spoken aloud |
| VAD unreliable in noisy rooms | Ship `push_to_talk=True` flag |
| Talkie occasionally uses post-1930 vocabulary | System-prompt + post-filter re-prompt |
| No Reachy hardware at dev time | MuJoCo simulator is the primary dev path |

---

## Credits

- **Talkie-1930** — Alec Radford, Nick Levine & David Duvenaud · Apache 2.0
- **Kokoro-82M** — hexgrad · Apache 2.0  
- **Reachy Mini SDK** — Pollen Robotics · Apache 2.0
- **faster-whisper** — SYSTRAN · MIT

---

## Licence

Apache 2.0 — see [LICENSE](LICENSE).

---

## Launch tweet

> The CTO of Hugging Face asked for a Reachy Mini app that speaks like Talkie-1930 in a posh pre-1931 British accent.
>
> Here it is. The robot uses Alec Radford's 13B vintage LM for its *words* and Kokoro for its *voice* — so it doesn't just sound 1930, it thinks 1930.
>
> Ask it about the wireless. Ask it about Twitter. Different vibes.
>
> Open-source, one-click install from the Reachy Mini app store.
>
> 🔗 Space: https://huggingface.co/spaces/09Catho/reachy-mini-talkie  
> 🔗 Repo: https://github.com/pollen-robotics/reachy-mini-talkie
>
> @ClementDelangue @Thom_Wolf @huggingface @pollenrobotics — thanks for the brief.
