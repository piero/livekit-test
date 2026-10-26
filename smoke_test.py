"""Scripted conversation against the real Ollama model, no console or LiveKit server.

Text only: the session has no STT/TTS, so it doesn't use LiveKit Inference minutes.
"""

import asyncio

from livekit.agents import AgentSession
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.plugins import openai

from agent import CLINIC, MODEL, Booking, Patient, Triage, check_insurance

TURNS = [
    "Hi, I'm Piero. I've had a sore knee for a week. My insurance is Unisalute.",
    "Thursday would be great.",
    "10:00 please.",
]


async def main() -> None:
    session = AgentSession[Patient](
        llm=openai.LLM.with_ollama(model=MODEL),
        userdata=Patient(),
        tools=[AsyncToolset(id="insurance", tools=[check_insurance])],
    )
    async with session:
        await session.start(Triage())
        for text in TURNS:
            print(f"\n> {text}")
            result = await session.run(user_input=text)
            for ev in result.events:
                print(f"  [{ev.type}] {ev.item}")
        await asyncio.sleep(10)  # let the async insurance check finish
        print("\nactive agent:", type(session.current_agent).__name__)
        print("userdata:", session.userdata)
        print("thursday still free:", CLINIC.free_slots("thursday"))
        assert isinstance(session.current_agent, Booking), "handoff did not happen"


asyncio.run(main())
