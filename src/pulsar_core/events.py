"""Kernel event envelope and payloads.

Every event flowing through the :class:`~pulsar_core.bus.EventBus` is an
immutable :class:`Event` carrying its own Asia/Shanghai timestamp and exactly
one typed payload. Market events, execution events and timer events share the
same envelope, so the loop orders them uniformly by ``(ts, publish sequence)``
regardless of who produced them — a replayed historical bar, a venue fill or
a scheduled callback.
"""

from __future__ import annotations

import enum
from datetime import datetime
from typing import Any, Mapping

from pydantic import Field, model_validator

from pulsar_contracts import Bar, ContractModel, ExecutionEvent, Snapshot, Timestamp

__all__ = ["EventKind", "SessionPhase", "TimerPayload", "Event"]


class EventKind(enum.StrEnum):
    """Kinds of events processed by the kernel loop."""

    SESSION = "session"  # run lifecycle (started / finished)
    MARKET = "market"  # bar or snapshot arrival
    EXECUTION = "execution"  # fill / accept / reject pushed by an execution port
    TIMER = "timer"  # scheduled callback


class SessionPhase(enum.StrEnum):
    """Phases of the run lifecycle reported through SESSION events."""

    STARTED = "started"
    FINISHED = "finished"


class TimerPayload(ContractModel):
    """Payload of a TIMER event: a named callback plus JSON-native data.

    Timer events are how anything inside the loop schedules future work
    (end-of-day hooks, rebalance ticks). The payload stays JSON-native so a
    full event journal remains serializable and comparable bit by bit.
    """

    name: str = Field(min_length=1)
    data: dict[str, Any] = Field(default_factory=dict)


_SLOT_BY_KIND: dict[EventKind, tuple[str, ...]] = {
    EventKind.SESSION: ("session_phase",),
    EventKind.MARKET: ("bar", "snapshot"),
    EventKind.EXECUTION: ("execution",),
    EventKind.TIMER: ("timer",),
}


class Event(ContractModel):
    """Immutable kernel event envelope; exactly one payload slot is set.

    The slot must match :attr:`kind`. The bus assigns dispatch order at
    publish time; the event itself never carries mutable sequence state.
    """

    kind: EventKind
    ts: Timestamp
    session_phase: SessionPhase | None = None
    bar: Bar | None = None
    snapshot: Snapshot | None = None
    execution: ExecutionEvent | None = None
    timer: TimerPayload | None = None

    @model_validator(mode="after")
    def _validate_payload(self) -> "Event":
        slots = ("session_phase", "bar", "snapshot", "execution", "timer")
        filled = [slot for slot in slots if getattr(self, slot) is not None]
        allowed = _SLOT_BY_KIND[self.kind]
        if len(filled) != 1:
            raise ValueError(
                f"{self.kind} event must carry exactly one payload, got {filled or 'none'}"
            )
        if filled[0] not in allowed:
            raise ValueError(
                f"payload slot '{filled[0]}' does not match event kind '{self.kind}' "
                f"(allowed: {list(allowed)})"
            )
        return self

    # -- typed constructors -------------------------------------------------

    @classmethod
    def session_started(cls, ts: datetime) -> "Event":
        return cls(kind=EventKind.SESSION, ts=ts, session_phase=SessionPhase.STARTED)

    @classmethod
    def session_finished(cls, ts: datetime) -> "Event":
        return cls(kind=EventKind.SESSION, ts=ts, session_phase=SessionPhase.FINISHED)

    @classmethod
    def of_bar(cls, bar: Bar) -> "Event":
        return cls(kind=EventKind.MARKET, ts=bar.ts, bar=bar)

    @classmethod
    def of_snapshot(cls, snapshot: Snapshot) -> "Event":
        return cls(kind=EventKind.MARKET, ts=snapshot.ts, snapshot=snapshot)

    @classmethod
    def of_execution(cls, execution: ExecutionEvent) -> "Event":
        return cls(kind=EventKind.EXECUTION, ts=execution.ts, execution=execution)

    @classmethod
    def timer_event(
        cls, when: datetime, name: str, data: Mapping[str, Any] | None = None
    ) -> "Event":
        return cls(
            kind=EventKind.TIMER,
            ts=when,
            timer=TimerPayload(name=name, data=dict(data) if data else {}),
        )
