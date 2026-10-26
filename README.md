# LiveKit Test

A test project for me to familiarize with [LiveKit Agents](https://docs.livekit.io/agents/).

It simulates a simple clinic front desk (agent), who takes the patient's name and reason and hands
them off to a booking agent.

What it tests:

- **Handoff**: `Triage` collects name and reason, then hands off to `Booking`, passing the
  chat history so nothing is asked twice.
- **Async tools**: `check_insurance` lives in a session-scoped `AsyncToolset`. It reports
  progress with `ctx.update()`, keeps running across the handoff, and its result is
  delivered to whichever agent is active when it finishes.
- **Shared state**: a `Patient` and a `Clinic` classes, read and written by tools.

## Run

Requires [uv](https://docs.astral.sh/uv/) and [Ollama](https://ollama.com/).

```bash
ollama pull qwen2.5:7b
uv sync
cp .env.example .env.local   # then fill in your LiveKit Cloud project's URL, key and secret
uv run agent.py console      # voice; add --text (or press Ctrl+T) to type instead
```

Speech runs on LiveKit Inference (Deepgram Flux for STT, Cartesia Sonic for TTS),
billed to a LiveKit Cloud project. The LLM runs on the local Ollama.

Try: *"Hi, I'm Tom, my knee hurts. My insurance is Unisalute."*.

Use a different model with `OLLAMA_MODEL=llama3.1 uv run agent.py console --text`.
The model must support tool calling.

## Scripted check

```bash
uv run python smoke_test.py
```

Plays a fixed three-turn conversation against the real model, prints every event
(messages, tool calls, handoff) and asserts that the handoff happened. *It doesn't
require any API keys.*
