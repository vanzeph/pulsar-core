"""Single-threaded deterministic event loop.

Design policy (core-engine design): determinism wins over throughput. Every
event — market, execution, timer, session — is dispatched by one thread in
strict ``(ts, publish_seq)`` order; ``publish_seq`` is a monotonically
increasing counter assigned at publish time, which gives same-timestamp
events a stable FIFO order. Multithreading and vectorized dispatch were
rejected because they break reproducibility; any future speedup happens
outside the kernel (feature pre-computation), never inside this loop.

Backtest and realtime runs share this loop unchanged: they differ only in
the :class:`~pulsar_core.clock.Clock` they install and in who publishes.
:meth:`run` drains the queue (backtest replay); :meth:`poll` dispatches only
events already due by the clock's current time (realtime).
"""

from __future__ import annotations

import heapq
import json
from collections import defaultdict
from datetime import datetime
from hashlib import sha256
from typing import Callable

from .clock import Clock
from .events import Event, EventKind

__all__ = ["EventBus", "Handler"]

#: A consumer of dispatched events; exceptions propagate and abort the run.
Handler = Callable[[Event], None]


class EventBus:
    """Timestamp-ordered single-threaded event bus.

    The bus owns the dispatch order (priority on ``(ts, publish_seq)``) and
    the journal of every event it has dispatched. Handlers may publish new
    events while being dispatched; the loop keeps draining until the queue
    is empty (or the caller's time window ends), so follow-up work always
    happens in the same deterministic pass.
    """

    def __init__(self, clock: Clock) -> None:
        self._clock = clock
        self._heap: list[tuple[datetime, int, Event]] = []
        self._publish_seq = 0
        self._kind_handlers: dict[EventKind, list[Handler]] = defaultdict(list)
        self._all_handlers: list[Handler] = []
        self._journal: list[Event] = []

    # -- wiring -------------------------------------------------------------

    @property
    def clock(self) -> Clock:
        """The time source this loop advances on every dispatch."""
        return self._clock

    @property
    def now(self) -> datetime:
        """Current time according to the installed clock."""
        return self._clock.now()

    def subscribe(self, kind: EventKind, handler: Handler) -> None:
        """Register ``handler`` for events of ``kind``.

        Handlers run in subscription order; kind-specific handlers run
        before wildcard handlers registered via :meth:`subscribe_all`.
        """
        self._kind_handlers[kind].append(handler)

    def subscribe_all(self, handler: Handler) -> None:
        """Register ``handler`` for every dispatched event."""
        self._all_handlers.append(handler)

    # -- publishing ---------------------------------------------------------

    def publish(self, event: Event) -> None:
        """Enqueue ``event``; dispatch order is ``(ts, publish_seq)``.

        Publishing is allowed from anywhere, including from inside a
        handler during dispatch.
        """
        self._publish_seq += 1
        heapq.heappush(self._heap, (event.ts, self._publish_seq, event))

    def schedule(
        self, when: datetime, name: str, data: dict[str, object] | None = None
    ) -> Event:
        """Publish a TIMER event due at ``when`` and return it."""
        event = Event.timer_event(when, name, data)
        self.publish(event)
        return event

    # -- dispatching --------------------------------------------------------

    def run(self, until: datetime | None = None) -> int:
        """Dispatch queued events in order; return how many were dispatched.

        Stops when the queue empties or when the next event's timestamp is
        ``>= until`` (exclusive bound — events beyond the window stay
        queued for a later pass). A handler exception propagates and
        terminates the run; the journal keeps every event dispatched so
        far, per the failure-behavior policy.
        """
        processed = 0
        while self._heap:
            ts, _, event = self._heap[0]
            if until is not None and ts >= until:
                break
            heapq.heappop(self._heap)
            self._dispatch(event)
            processed += 1
        return processed

    def poll(self) -> int:
        """Realtime counterpart of :meth:`run`.

        Dispatch every event whose timestamp is due by the clock's current
        time and return; the caller (a realtime session) calls again as new
        events arrive. Same ordering guarantees as :meth:`run`.
        """
        return self.run(until=self._clock.now())

    @property
    def pending(self) -> int:
        """Number of events queued but not yet dispatched."""
        return len(self._heap)

    # -- journal ------------------------------------------------------------

    @property
    def journal(self) -> tuple[Event, ...]:
        """Every dispatched event, in dispatch order."""
        return tuple(self._journal)

    @property
    def journal_digest(self) -> str:
        """SHA-256 over the canonical serialization of the whole journal.

        Two runs with identical inputs must produce the same digest — this
        is the bit-level identity check of the reproducibility promise.
        """
        digest = sha256()
        for event in self._journal:
            digest.update(canonical_event_json(event).encode("utf-8"))
            digest.update(b"\n")
        return digest.hexdigest()

    # -- internals ----------------------------------------------------------

    def _dispatch(self, event: Event) -> None:
        self._clock.advance_to(event.ts)
        self._journal.append(event)
        for handler in tuple(self._kind_handlers[event.kind]):
            handler(event)
        for handler in tuple(self._all_handlers):
            handler(event)


def canonical_event_json(event: Event) -> str:
    """Canonical JSON form of an event: sorted keys, no whitespace.

    Used for digests and for comparing journals across runs. Pydantic's JSON
    mode gives deterministic representations for the domain payloads; sorting
    keys removes any reliance on field ordering.
    """
    return json.dumps(
        event.model_dump(mode="json"),
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=False,
        allow_nan=False,
    )
