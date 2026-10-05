"""Experiment lifecycle (上下线): status machine, assembly gate, retire-stop.

Core-engine design, 模型配置生命周期:

* experiment documents carry a required ``status`` (candidate / active /
  retired) resolving to the modes that status admits;
* the assembly validator rejects every status/mode pairing the design's
  table does not admit — Paper / Live assemble ``active`` only, and a
  retired experiment runs nowhere (read-only post-mortem);
* retire/activate are human-driven registry transitions (the experiments
  directory under git IS the registry) that rewrite exactly the status
  line and return an audit record;
* a running session that receives a retire immediately stops producing
  new order intents — existing positions are left to the strategy's own
  exit rules — and the action is archived into the run's RunManifest;
* every assembly records the git commit of the configuration it used.
"""

from __future__ import annotations

import shutil
import subprocess
from datetime import date, datetime

import pytest
from pulsar_contracts import SHANGHAI_TZ, Side

from conftest import FillingExecutionPort, ScriptedClosesPort, down_up_down

from pulsar_core import (
    RUN_MODES,
    ExperimentConfig,
    ExperimentConfigError,
    LifecycleError,
    LifecycleRecord,
    Params,
    ReplaySession,
    STATUS_ALLOWED_MODES,
    StrategyBase,
    StrategyRuntime,
    RunManifest,
    activate_experiment,
    experiment_commit,
    load_experiment,
    parse_experiment,
    retire_experiment,
    run_experiment,
    validate_assembly,
)
from pulsar_core.bus import EventBus
from pulsar_core.clock import BacktestClock
from pulsar_core.events import EventKind
from pulsar_core.manifest import MODES
from pulsar_core.session import _day_start

BASE_TOML = """
[experiment]
id = "lifecycle_case"{extra}

[universe]
symbols = ["UP"]

[factors]
names = ["momentum_20"]
preprocess = ["winsorize", "zscore"]

[model]
type = "equal_weight"

[portfolio]
method = "top_n"
top_n = 1
rebalance = "monthly"

[backtest]
start = 2026-06-01
end = 2026-07-31
costs = "none"
seed = 5
"""


def write_config(tmp_path, status: str | None, *, name: str = "experiment.toml"):
    """Write an experiment document with the given (or absent) status."""
    extra = f'\nstatus = "{status}"' if status is not None else ""
    path = tmp_path / name
    path.write_text(BASE_TOML.format(extra=extra), encoding="utf-8")
    return path


def make_config(tmp_path, status: str) -> ExperimentConfig:
    return load_experiment(write_config(tmp_path, status))


class TestStatusFieldLoading:
    def test_missing_status_is_rejected(self, tmp_path) -> None:
        with pytest.raises(ExperimentConfigError, match="experiment.status is required"):
            load_experiment(write_config(tmp_path, None))

    def test_non_string_status_is_rejected(self, tmp_path) -> None:
        path = tmp_path / "experiment.toml"
        path.write_text(
            BASE_TOML.format(extra="\nstatus = 42"), encoding="utf-8"
        )
        with pytest.raises(ExperimentConfigError, match="must be a string"):
            load_experiment(path)

    def test_illegal_status_is_rejected_listing_the_three(self, tmp_path) -> None:
        with pytest.raises(ExperimentConfigError, match="candidate.*active.*retired"):
            make_config(tmp_path, "running")

    @pytest.mark.parametrize("status", ["candidate", "active", "retired"])
    def test_legal_statuses_resolve_with_their_allowed_modes(
        self, tmp_path, status: str
    ) -> None:
        config = make_config(tmp_path, status)
        assert config.status == status
        assert config.allowed_modes == STATUS_ALLOWED_MODES[status]

    def test_allowed_modes_follow_the_design_table(self) -> None:
        assert STATUS_ALLOWED_MODES["candidate"] == ("research",)
        assert STATUS_ALLOWED_MODES["active"] == ("research", "paper", "live")
        assert STATUS_ALLOWED_MODES["retired"] == ()

    def test_status_is_part_of_the_config_snapshot(self, tmp_path) -> None:
        snapshot = make_config(tmp_path, "active").config_snapshot()
        assert snapshot["experiment"]["status"] == "active"

    def test_allowed_modes_is_not_declarable_in_toml(self, tmp_path) -> None:
        path = tmp_path / "experiment.toml"
        path.write_text(
            BASE_TOML.format(extra='\nstatus = "candidate"\nallowed_modes = ["live"]'),
            encoding="utf-8",
        )
        with pytest.raises(ExperimentConfigError, match="unknown key.*experiment"):
            load_experiment(path)


class TestAssemblyValidation:
    def test_candidate_runs_research_only(self, tmp_path) -> None:
        config = make_config(tmp_path, "candidate")
        validate_assembly("research", config)  # allowed, no raise
        for mode in ("paper", "live"):
            with pytest.raises(LifecycleError) as excinfo:
                validate_assembly(mode, config)
            message = str(excinfo.value)
            assert "candidate" in message and mode in message
            assert "research" in message  # the reason states what is allowed
            assert excinfo.value.status == "candidate"
            assert excinfo.value.mode == mode

    def test_active_runs_every_mode(self, tmp_path) -> None:
        config = make_config(tmp_path, "active")
        for mode in ("research", "paper", "live"):
            validate_assembly(mode, config)

    def test_retired_runs_nowhere(self, tmp_path) -> None:
        config = make_config(tmp_path, "retired")
        for mode in ("research", "paper", "live"):
            with pytest.raises(LifecycleError) as excinfo:
                validate_assembly(mode, config)
            message = str(excinfo.value)
            assert "retired" in message and mode in message
            assert "post-mortem" in message

    def test_unknown_mode_is_rejected(self, tmp_path) -> None:
        config = make_config(tmp_path, "active")
        with pytest.raises(LifecycleError, match="unknown run mode"):
            validate_assembly("imagination", config)

    def test_run_modes_mirror_the_manifest_modes(self) -> None:
        assert RUN_MODES == MODES


class TestRegistryTransitions:
    def test_activate_rewrites_status_and_returns_the_audit_record(
        self, tmp_path
    ) -> None:
        path = write_config(tmp_path, "candidate")
        before = path.read_text(encoding="utf-8")
        stamp = datetime(2026, 10, 1, 9, 30, tzinfo=SHANGHAI_TZ)
        record = activate_experiment(
            path,
            reason="passed research acceptance",
            operator="guanlan",
            confirmed=True,
            ts=stamp,
            commit="0f1e2d3c4b5a69788",
        )
        assert record == LifecycleRecord(
            action="activate",
            from_status="candidate",
            to_status="active",
            reason="passed research acceptance",
            operator="guanlan",
            ts="2026-10-01T09:30:00+08:00",
            config_commit="0f1e2d3c4b5a69788",
        )
        after = path.read_text(encoding="utf-8")
        assert 'status = "active"' in after
        # minimal-diff rewrite: exactly the status line changed
        assert after == before.replace('status = "candidate"', 'status = "active"')
        assert load_experiment(path).status == "active"

    def test_activate_requires_explicit_human_confirmation(self, tmp_path) -> None:
        path = write_config(tmp_path, "candidate")
        with pytest.raises(LifecycleError, match="human confirmation"):
            activate_experiment(path, reason="r", operator="guanlan")
        assert load_experiment(path).status == "candidate"  # untouched

    def test_activate_from_active_is_illegal(self, tmp_path) -> None:
        path = write_config(tmp_path, "active")
        with pytest.raises(LifecycleError, match="illegal transition"):
            activate_experiment(
                path, reason="r", operator="guanlan", confirmed=True
            )

    def test_retire_rewrites_status_and_records_reason_operator_ts(
        self, tmp_path
    ) -> None:
        path = write_config(tmp_path, "active")
        stamp = datetime(2026, 10, 5, 15, 0, tzinfo=SHANGHAI_TZ)
        record = retire_experiment(
            path,
            reason="signal decayed after regime change",
            operator="vanzeph",
            ts=stamp,
            commit="abcdef1234567890abcdef1234567890abcdef12",
        )
        assert record.action == "retire"
        assert (record.from_status, record.to_status) == ("active", "retired")
        assert record.reason == "signal decayed after regime change"
        assert record.operator == "vanzeph"
        assert record.ts == "2026-10-05T15:00:00+08:00"
        assert record.config_commit == "abcdef1234567890abcdef1234567890abcdef12"
        assert load_experiment(path).status == "retired"

    def test_retire_defaults_the_timestamp_to_wall_clock(self, tmp_path) -> None:
        path = write_config(tmp_path, "active")
        record = retire_experiment(path, reason="r", operator="guanlan")
        parsed = datetime.fromisoformat(record.ts)
        assert parsed.tzinfo is not None
        assert abs((parsed - datetime.now(SHANGHAI_TZ)).total_seconds()) < 60

    def test_retire_from_candidate_is_illegal(self, tmp_path) -> None:
        path = write_config(tmp_path, "candidate")
        with pytest.raises(LifecycleError, match="illegal transition") as excinfo:
            retire_experiment(path, reason="r", operator="guanlan")
        assert excinfo.value.status == "candidate"
        assert load_experiment(path).status == "candidate"  # untouched

    def test_double_retire_is_illegal(self, tmp_path) -> None:
        path = write_config(tmp_path, "active")
        retire_experiment(path, reason="first", operator="guanlan")
        with pytest.raises(LifecycleError, match="illegal transition"):
            retire_experiment(path, reason="second", operator="guanlan")

    @pytest.mark.parametrize("bad_reason", ["", "   "])
    def test_empty_reason_or_operator_is_rejected(
        self, tmp_path, bad_reason: str
    ) -> None:
        path = write_config(tmp_path, "active")
        with pytest.raises(LifecycleError, match="non-empty reason"):
            retire_experiment(path, reason=bad_reason, operator="guanlan")
        with pytest.raises(LifecycleError, match="non-empty operator"):
            retire_experiment(path, reason="valid", operator=bad_reason)

    def test_rewrite_preserves_indentation_and_trailing_comment(
        self, tmp_path
    ) -> None:
        path = tmp_path / "experiment.toml"
        path.write_text(
            BASE_TOML.format(extra='\n  status = "candidate"  # promoted by ops'),
            encoding="utf-8",
        )
        activate_experiment(
            path, reason="r", operator="guanlan", confirmed=True
        )
        text = path.read_text(encoding="utf-8")
        assert '  status = "active"  # promoted by ops' in text


class TestExperimentCommit:
    def test_explicit_commit_wins_and_is_normalized(self, tmp_path) -> None:
        path = write_config(tmp_path, "candidate")
        assert (
            experiment_commit(path, commit="0F1E2D3C4B5A69788")
            == "0f1e2d3c4b5a69788"
        )

    @pytest.mark.parametrize("bad", ["xyz", "abc", "", "0123456789abcdef0123456789abcdef0123456789"])
    def test_explicit_commit_must_look_like_a_sha(self, tmp_path, bad: str) -> None:
        path = write_config(tmp_path, "candidate")
        with pytest.raises(LifecycleError, match="not a git sha"):
            experiment_commit(path, commit=bad)

    def test_file_outside_any_repo_resolves_unknown(self, tmp_path) -> None:
        assert experiment_commit(write_config(tmp_path, "candidate")) == "unknown"

    def test_head_commit_of_the_owning_repo_is_resolved(self, tmp_path) -> None:
        if shutil.which("git") is None:  # pragma: no cover - CI images carry git
            pytest.skip("git binary not available")
        repo = tmp_path / "registry"
        experiments = repo / "experiments"
        experiments.mkdir(parents=True)
        path = experiments / "experiment.toml"
        path.write_text(BASE_TOML.format(extra='\nstatus = "candidate"'), "utf-8")
        for argv in (
            ["git", "init", "-q", str(repo)],
            ["git", "-C", str(repo), "config", "user.email", "ops@example.com"],
            ["git", "-C", str(repo), "config", "user.name", "ops"],
            ["git", "-C", str(repo), "add", "experiments/experiment.toml"],
            ["git", "-C", str(repo), "commit", "-q", "-m", "register experiment"],
        ):
            subprocess.run(argv, check=True, capture_output=True, timeout=30)
        resolved = experiment_commit(path)
        head = subprocess.run(
            ["git", "-C", str(repo), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=30,
        ).stdout.strip()
        assert resolved == head


class TestManifestLifecycleOverlay:
    def test_overlay_fields_do_not_fork_run_ids(self) -> None:
        base = RunManifest.build(
            mode="paper", seed=1, code_version="c0ffee", config={"a": 1}
        )
        annotated = RunManifest.build(
            mode="paper",
            seed=1,
            code_version="c0ffee",
            config={"a": 1},
            config_commit="abcdef1234567",
            lifecycle=[
                LifecycleRecord(
                    action="retire",
                    from_status="active",
                    to_status="retired",
                    reason="decay",
                    operator="ops",
                    ts=datetime(2026, 10, 5, 15, 0, tzinfo=SHANGHAI_TZ),
                )
            ],
        )
        assert annotated.run_id == base.run_id
        assert annotated.config_commit == "abcdef1234567"
        assert annotated.lifecycle[0].action == "retire"
        assert annotated.lifecycle[0].ts == "2026-10-05T15:00:00+08:00"

    def test_record_lifecycle_appends_in_place_and_round_trips(self, tmp_path) -> None:
        manifest = RunManifest.build(mode="live", seed=2, code_version="c0ffee")
        record = LifecycleRecord(
            action="retire",
            from_status="active",
            to_status="retired",
            reason="halt",
            operator="ops",
            ts="2026-10-05T15:00:00+08:00",
        )
        manifest.record_lifecycle(record)
        manifest.record_lifecycle(record.model_copy())
        assert len(manifest.lifecycle) == 2
        path = manifest.write(tmp_path / "run_manifest.json")
        loaded = RunManifest.model_validate_json(path.read_text("utf-8"))
        assert loaded.lifecycle == manifest.lifecycle
        assert loaded.config_commit is None

    def test_record_accepts_datetime_ts_and_validates_commit_shape(self) -> None:
        record = LifecycleRecord(
            action="activate",
            from_status="candidate",
            to_status="active",
            reason="ok",
            operator="ops",
            ts=datetime(2026, 10, 1, 9, 30, tzinfo=SHANGHAI_TZ),
        )
        assert record.ts == "2026-10-01T09:30:00+08:00"
        with pytest.raises(ValueError):
            LifecycleRecord(
                action="activate",
                from_status="candidate",
                to_status="active",
                reason="ok",
                operator="ops",
                ts="2026-10-01T09:30:00+08:00",
                config_commit="not-a-sha",
            )


# -- runtime retire checkpoint -------------------------------------------------

PATH = down_up_down(12, 18, 12)  # fall -> rally (golden cross) -> slide (exit)


class DualMA(StrategyBase):
    """Reference strategy copy: declares targets on MA crossings."""

    params = Params(fast=3, slow=6)

    def on_bar(self, ctx) -> None:
        fast = ctx.sma("close", self.params.fast)
        slow = ctx.sma("close", self.params.slow)
        if ctx.cross_up(fast, slow):
            ctx.target_weight(ctx.symbol, 1.0)
        elif ctx.cross_down(fast, slow):
            ctx.target_weight(ctx.symbol, 0.0)


def build_run():
    """One dual-MA replay session over the scripted path."""
    port = ScriptedClosesPort({"600000": PATH})
    start, end = port.span()
    bus = EventBus(BacktestClock(_day_start(start)))
    venue = FillingExecutionPort(now=lambda: bus.now)
    runtime = StrategyRuntime(
        bus=bus, port=venue, strategy=DualMA(), initial_cash=100_000.0
    )
    session = ReplaySession(
        port=port,
        symbols=["600000"],
        start=start,
        end=end,
        seed=3,
        config={"strategy": {"params": dict(fast=3, slow=6)}},
        bus=bus,
        on_manifest=runtime.bind_manifest,
    )
    return session, runtime, venue


class TestRuntimeRetireCheckpoint:
    def test_control_run_buys_then_sells(self) -> None:
        session, runtime, _ = build_run()
        run = session.run()
        sides = [intent.side for intent in runtime.intents]
        assert Side.BUY in sides and Side.SELL in sides
        assert run.manifest.lifecycle == []

    def test_retire_mid_run_stops_new_intents_and_leaves_manifest_trace(
        self,
    ) -> None:
        session, runtime, venue = build_run()
        frozen: list[int] = []

        def retire_once_the_position_exists(event) -> None:
            # subscribed after the runtime, so it observes the same bar's
            # pipeline output; retire lands between this bar and the next
            if venue.submitted and not runtime.retired:
                runtime.retire(
                    reason="signal decayed after regime change",
                    operator="guanlan",
                )
                frozen.append(len(runtime.intents))

        session.bus.subscribe(EventKind.MARKET, retire_once_the_position_exists)
        run = session.run()

        # the session kept running to completion, but produced no new
        # intent after the retire: the control run's SELL never arrived
        assert frozen and frozen[0] >= 1
        assert len(runtime.intents) == frozen[0]
        assert all(intent.side is Side.BUY for intent in runtime.intents)
        assert len(venue.submitted) == frozen[0]
        assert runtime.retired
        assert runtime.retire_record is not None
        assert runtime.retire_record.reason == "signal decayed after regime change"
        assert runtime.retire_record.operator == "guanlan"

        # existing positions are NOT force-sold: the engine leaves them to
        # the strategy's own exit rules
        assert [view.symbol for view in runtime.account.positions()] == ["600000"]

        # the manifest archives the lifecycle trace (status change,
        # reason, timestamp) and the run id itself is untouched by it
        records = run.manifest.lifecycle
        assert len(records) == 1
        assert records[0].action == "retire"
        assert (records[0].from_status, records[0].to_status) == ("active", "retired")
        assert records[0].reason == "signal decayed after regime change"
        datetime.fromisoformat(records[0].ts)  # a real ISO timestamp
        control_session, control_runtime, _ = build_run()
        control_run = control_session.run()
        assert control_runtime.intents  # the control really did trade
        assert run.run_id == control_run.run_id

    def test_double_retire_is_idempotent_and_keeps_the_first_record(self) -> None:
        session, runtime, venue = build_run()
        retired_once: list[bool] = []

        def retire_twice_after_the_first_fill(event) -> None:
            if venue.submitted and not retired_once:
                retired_once.append(True)
                first = runtime.retire(reason="first", operator="ops")
                second = runtime.retire(reason="second", operator="ops")
                assert second == first  # idempotent: the first record wins

        session.bus.subscribe(EventKind.MARKET, retire_twice_after_the_first_fill)
        run = session.run()
        assert retired_once
        assert len(run.manifest.lifecycle) == 1
        assert run.manifest.lifecycle[0].reason == "first"

    def test_retire_before_any_intent_blocks_everything(self) -> None:
        session, runtime, venue = build_run()

        def retire_immediately(event) -> None:
            if not runtime.retired:
                runtime.retire(reason="pre-market halt", operator="ops")

        session.bus.subscribe(EventKind.SESSION, retire_immediately)
        session.run()
        assert runtime.intents == ()
        assert venue.submitted == []
        assert runtime.retired


# -- experiment runner integration ----------------------------------------------


class TestRunnerEnforcesLifecycle:
    def make_experiment_port(self):
        return ScriptedClosesPort(
            {"UP": [100.0 * 1.01**i for i in range(50)]}
        )

    def test_candidate_config_runs_in_research(self, tmp_path) -> None:
        config = make_config(tmp_path, "candidate")
        port = self.make_experiment_port()
        # the fixture window (June-July 2026) is inside the scripted days
        result = run_experiment(
            config, port=port, venue=lambda bus: FillingExecutionPort(
                now=lambda: bus.now
            )
        )
        assert result.experiment_id == "lifecycle_case"
        assert result.manifest.mode == "research"
        assert result.manifest.lifecycle == []
        assert result.runtime.intents  # the July rebalance really traded

    def test_retired_config_is_refused_by_the_research_runner(self, tmp_path) -> None:
        config = make_config(tmp_path, "retired")
        with pytest.raises(LifecycleError, match="retired") as excinfo:
            run_experiment(
                config,
                port=self.make_experiment_port(),
                venue=lambda bus: FillingExecutionPort(now=lambda: bus.now),
            )
        assert excinfo.value.status == "retired"
        assert excinfo.value.mode == "research"

    def test_explicit_config_commit_is_stamped_into_the_manifest(
        self, tmp_path
    ) -> None:
        config = make_config(tmp_path, "candidate")
        port = self.make_experiment_port()

        def make_venue(bus):
            return FillingExecutionPort(now=lambda: bus.now)

        first = run_experiment(
            config, port=port, venue=make_venue, config_commit="abcdef1234567890"
        )
        assert first.manifest.config_commit == "abcdef1234567890"

        # provenance annotations never fork run ids: same inputs, same id
        second = run_experiment(
            config, port=port, venue=make_venue, config_commit="abcdef1234567890"
        )
        assert second.run_id == first.run_id

    def test_in_memory_configs_carry_no_source_path(self, tmp_path) -> None:
        tree = {
            "experiment": {
                "id": "mem",
                "status": "candidate",
            },
            "universe": {"symbols": ["600000"]},
            "factors": {"names": ["momentum_20"], "preprocess": []},
            "model": {"type": "equal_weight"},
            "portfolio": {"method": "top_n", "top_n": 1, "rebalance": "monthly"},
            "backtest": {
                "start": date(2026, 6, 1),
                "end": date(2026, 6, 30),
                "costs": "none",
                "seed": 1,
            },
        }
        config = parse_experiment(tree)
        assert config.source_path is None
        assert config.status == "candidate"
        loaded = load_experiment(write_config(tmp_path, "candidate"))
        assert loaded.source_path is not None
