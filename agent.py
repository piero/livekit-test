"""Clinic front desk: two agents with a handoff and a long-running async tool.

- Triage greets the patient and starts an insurance check, then hands off to Booking.
- The insurance check lives in a session-scoped AsyncToolset, so it keeps running
  across the handoff and reports its result to whichever agent is active.
- The LLM runs locally via Ollama; speech (STT/TTS) runs on LiveKit Inference.
"""

import asyncio
import os
from dataclasses import dataclass

from dotenv import load_dotenv
from livekit.agents import (
    Agent,
    AgentServer,
    AgentSession,
    JobContext,
    RunContext,
    cli,
    function_tool,
)
from livekit.agents.llm.async_toolset import AsyncToolset
from livekit.plugins import openai

load_dotenv(".env.local")  # Required: LIVEKIT_URL, LIVEKIT_API_KEY, LIVEKIT_API_SECRET

MODEL = os.environ.get("OLLAMA_MODEL", "qwen2.5:7b")


@dataclass
class Patient:
    name: str | None = None
    reason: str | None = None
    booked_slot: str | None = None
    insurance: str | None = None


class Clinic:
    """The clinic's schedule, shared by all sessions in this process.

    Stand-in for a real scheduling backend: in production this would be a
    database or booking API, since each call may run in its own process.
    """

    def __init__(self) -> None:
        self._slots: dict[str, list[str]] = {
            "monday": ["09:00", "11:30"],
            "tuesday": ["14:00"],
            "wednesday": [],
            "thursday": ["10:00", "16:30"],
            "friday": ["09:30"],
        }
        self._lock = asyncio.Lock()

    def free_slots(self, day: str) -> list[str] | None:
        """Free times for a weekday, or None if the clinic is closed."""
        free = self._slots.get(day.lower())
        return None if free is None else list(free)

    async def book(self, day: str, time: str) -> bool:
        """Take a slot; False if it's no longer free."""
        async with self._lock:
            free = self._slots.get(day.lower(), [])
            if time not in free:
                return False
            free.remove(time)
            return True


CLINIC = Clinic()


@function_tool
async def check_insurance(ctx: RunContext[Patient], provider: str) -> str:
    """Check the patient's insurance coverage with their provider. Takes a while.

    Args:
        provider: Name of the insurance provider, e.g. "Unisalute".
    """
    await ctx.update(f"Contacting {provider}, this can take a few seconds...")
    await asyncio.sleep(4)
    await ctx.update("Provider reached, verifying the policy...")
    await asyncio.sleep(4)
    ctx.userdata.insurance = f"{provider}: covered, 20 EUR co-pay"
    return ctx.userdata.insurance


class Triage(Agent):
    def __init__(self) -> None:
        super().__init__(
            instructions=(
                "You are the front desk of a small clinic. Be brief and friendly. "
                "Ask the patient for their name and the reason for the visit. "
                "If they mention an insurance provider, call check_insurance once; "
                "it runs in the background, so do not wait for it and never call it again. "
                "As soon as you know name and reason, call transfer_to_booking."
            )
        )

    @function_tool
    async def transfer_to_booking(
        self, ctx: RunContext[Patient], name: str, reason: str
    ) -> tuple[Agent, str]:
        """Hand the patient over to the booking assistant.

        Args:
            name: The patient's name.
            reason: Short reason for the visit.
        """
        ctx.userdata.name = name
        ctx.userdata.reason = reason
        # carry the conversation over, so Booking knows what was already said
        return Booking(CLINIC, chat_ctx=self.chat_ctx), f"Transferring {name} to booking."


class Booking(Agent):
    def __init__(self, clinic: Clinic, chat_ctx=None) -> None:
        self.clinic = clinic
        super().__init__(
            chat_ctx=chat_ctx,
            instructions=(
                "You book clinic appointments. Be brief. "
                "Use list_slots to see availability for a day, then book_slot to confirm. "
                "Never invent times: only offer slots returned by list_slots."
            )
        )

    async def on_enter(self) -> None:
        p: Patient = self.session.userdata
        await self.session.generate_reply(
            instructions=(
                f"Greet {p.name}, who needs a visit for: {p.reason}. "
                "If they already said which day they prefer, call list_slots for it; "
                "otherwise ask which day suits them."
            )
        )

    @function_tool
    async def list_slots(self, ctx: RunContext[Patient], day: str) -> str:
        """List free appointment times for a weekday.

        Args:
            day: Weekday name, e.g. "monday".
        """
        free = self.clinic.free_slots(day)
        if free is None:
            return "The clinic is closed that day."
        return ", ".join(free) if free else "No free slots that day."

    @function_tool
    async def book_slot(self, ctx: RunContext[Patient], day: str, time: str) -> str:
        """Book an appointment.

        Args:
            day: Weekday name, e.g. "monday".
            time: Time as HH:MM, one of the slots returned by list_slots.
        """
        if not await self.clinic.book(day, time):
            free = self.clinic.free_slots(day) or []
            return f"{time} is not available on {day}. Free: {', '.join(free) or 'none'}."
        ctx.userdata.booked_slot = f"{day} {time}"
        return f"Booked {ctx.userdata.name} on {day} at {time}."


server = AgentServer()


@server.rtc_session()
async def entrypoint(ctx: JobContext) -> None:
    session = AgentSession[Patient](
        stt="deepgram/flux-general-en",
        llm=openai.LLM.with_ollama(model=MODEL),
        tts="cartesia/sonic-3.6",
        userdata=Patient(),
        tools=[AsyncToolset(id="insurance", tools=[check_insurance])],
    )
    await session.start(agent=Triage(), room=ctx.room)
    await session.generate_reply(instructions="Greet the patient and ask how you can help.")


if __name__ == "__main__":
    cli.run_app(server)
