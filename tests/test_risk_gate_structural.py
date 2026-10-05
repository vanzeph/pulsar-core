"""Structural tests: the risk exit cannot be bypassed by a strategy.

Two complementary angles (core-engine design: "风控链在意图出口处强制执行,
策略无法绕过"):

1. **Source-level**: an AST scan proves ``.submit(...)`` — the only way to
   reach a venue — appears in exactly one module of the package, the
   pipeline runtime, and that the strategy-facing modules never even name
   the execution machinery.
2. **Object-level**: contexts handed to strategies expose no attribute
   that is (or leads to, within one hop) an execution port, a risk gate
   or the runtime; the account is visible only as an immutable snapshot.
3. **Behavioral**: a gate whose every rule rejects still swallows every
   order even though the strategy declared a full-position target — and
   the rejections are recorded with reasons.
"""

from __future__ import annotations

import ast
import sys
from datetime import date
from pathlib import Path

import pytest
from pulsar_contracts import ExecutionPort

from conftest import FillingExecutionPort, ScriptedClosesPort, down_up_down

from pulsar_core import (
    BacktestClock,
    EventBus,
    OrderDraft,
    Params,
    ReplaySession,
    RiskGate,
    RiskRule,
    StrategyBase,
    StrategyRuntime,
)
from pulsar_core.session import _day_start

SRC_ROOT = Path(__file__).resolve().parents[1] / "src" / "pulsar_core"


# -- 1. source-level AST guarantees ----------------------------------------------


def _submit_call_sites() -> dict[str, int]:
    """Count ``.submit(...)`` attribute calls per module of the package."""
    sites: dict[str, int] = {}
    for path in sorted(SRC_ROOT.rglob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
        count = 0
        for node in ast.walk(tree):
            if (
                isinstance(node, ast.Call)
                and isinstance(node.func, ast.Attribute)
                and node.func.attr == "submit"
            ):
                count += 1
        if count:
            sites[path.name] = count
    return sites


def test_submit_is_called_only_inside_the_pipeline_runtime() -> None:
    sites = _submit_call_sites()
    assert sites == {"runtime.py": 1}, (
        f"execution-port submissions must exist only in runtime.py "
        f"(the post-gate emitter); found {sites}"
    )


def test_strategy_modules_never_name_the_execution_machinery() -> None:
    """The strategy-facing source never mentions orders, ports or gates."""
    for module in ("strategy.py", "signals.py", "params.py", "account.py"):
        source = (SRC_ROOT / module).read_text(encoding="utf-8")
        for forbidden in ("OrderIntent", "ExecutionPort", "RiskGate", "submit"):
            assert forbidden not in source, (
                f"{module} must not reference {forbidden}: strategy-facing "
                "code has no ordering surface"
            )


# -- 2. object-level reachability -----------------------------------------------


class _BuyEverything(StrategyBase):
    params = Params()

    def on_bar(self, ctx) -> None:
        ctx.target_weight(ctx.symbol, 1.0)
        # stash the context so the test can interrogate it afterwards
        _BuyEverything.seen_contexts.append(ctx)


_BuyEverything.seen_contexts = []


def _run_once() -> tuple:
    closes = down_up_down(6, 8, 6)
    port = ScriptedClosesPort({"600000": closes})
    start, end = port.span()
    bus = EventBus(BacktestClock(_day_start(start)))
    venue = FillingExecutionPort(now=lambda: bus.now)
    runtime = StrategyRuntime(
        bus=bus, port=venue, strategy=_BuyEverything(), initial_cash=50_000.0
    )
    session = ReplaySession(
        port=port,
        symbols=["600000"],
        start=start,
        end=end,
        seed=1,
        bus=bus,
        on_manifest=runtime.bind_manifest,
    )
    session.run()
    return runtime, venue


def test_contexts_expose_no_ordering_surface() -> None:
    runtime, venue = _run_once()
    assert _BuyEverything.seen_contexts, "strategy must have received contexts"
    for ctx in _BuyEverything.seen_contexts:
        # no attribute is or provides a submit/order entry point
        for name, value in vars(ctx).items():
            assert not hasattr(value, "submit"), f"ctx.{name} can submit orders"
            assert not hasattr(value, "cancel"), f"ctx.{name} can cancel orders"
            assert not hasattr(value, "review"), f"ctx.{name} reaches the gate"
            assert not isinstance(value, ExecutionPort), f"ctx.{name} is a port"
        # the contexts themselves have no such methods either
        for forbidden in ("submit", "cancel", "order", "place_order", "amend"):
            assert not hasattr(ctx, forbidden)


def test_contexts_hold_no_reference_to_the_engine() -> None:
    runtime, venue = _run_once()
    for ctx in _BuyEverything.seen_contexts:
        for name, value in vars(ctx).items():
            assert not isinstance(value, StrategyRuntime), f"ctx.{name} is the runtime"
            assert not isinstance(value, RiskGate), f"ctx.{name} is the gate"
            assert not isinstance(value, EventBus), f"ctx.{name} is the bus"
        # the account is visible only as an immutable snapshot
        snapshot = ctx.portfolio
        with pytest.raises(Exception):
            snapshot.cash = 1_000_000.0  # type: ignore[misc] - frozen model


# -- 3. behavioral: the gate is the only exit ------------------------------------


class _RejectAll(RiskRule):
    name = "reject_all_for_test"

    def check(self, draft: OrderDraft, view) -> str:
        return "blocked by test rule"


def test_a_fully_rejecting_gate_blocks_every_order() -> None:
    closes = down_up_down(6, 8, 6)
    port = ScriptedClosesPort({"600000": closes})
    start, end = port.span()
    bus = EventBus(BacktestClock(_day_start(start)))
    venue = FillingExecutionPort(now=lambda: bus.now)
    runtime = StrategyRuntime(
        bus=bus,
        port=venue,
        strategy=_BuyEverything(),
        gate=RiskGate((_RejectAll(),)),
        initial_cash=50_000.0,
    )
    session = ReplaySession(
        port=port,
        symbols=["600000"],
        start=start,
        end=end,
        seed=1,
        bus=bus,
        on_manifest=runtime.bind_manifest,
    )
    session.run()

    # the strategy declared a 100% target on every bar, the pipeline sized
    # drafts, and the venue still received nothing at all
    assert venue.submitted == []
    assert runtime.intents == ()
    assert runtime.submissions == []
    assert runtime.rejections, "rejections must be recorded with reasons"
    assert all(record.rule == "reject_all_for_test" for record in runtime.rejections)
    assert all(record.reason == "blocked by test rule" for record in runtime.rejections)
    # nothing was bought, so the account is untouched
    assert runtime.account.snapshot().positions == ()
    assert runtime.account.cash == 50_000.0


def test_strategy_exception_aborts_the_run_and_keeps_evidence() -> None:
    class Exploding(StrategyBase):
        params = Params()

        def on_bar(self, ctx) -> None:
            raise RuntimeError("strategy bug")

    closes = down_up_down(6, 8, 6)
    port = ScriptedClosesPort({"600000": closes})
    start, end = port.span()
    bus = EventBus(BacktestClock(_day_start(start)))
    venue = FillingExecutionPort(now=lambda: bus.now)
    runtime = StrategyRuntime(bus=bus, port=venue, strategy=Exploding())
    session = ReplaySession(
        port=port,
        symbols=["600000"],
        start=start,
        end=end,
        seed=1,
        bus=bus,
        on_manifest=runtime.bind_manifest,
    )
    with pytest.raises(RuntimeError, match="strategy bug"):
        session.run()
    # the journal preserves everything dispatched before the failure
    assert bus.journal, "events produced before the abort stay inspectable"
    assert venue.submitted == []


def test_side_is_not_reachable_as_a_bypass_via_state() -> None:
    # belt and braces: the explicit state dict stays plain data
    runtime, venue = _run_once()
    assert isinstance(runtime.state, dict)
    import json

    json.dumps(runtime.state, default=str)  # JSON-native by convention
