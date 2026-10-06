"""Torch-free behaviors of the ML surface: run everywhere, no torch needed.

The [ml] extra contract: a default (torch-free) install keeps
``import pulsar_core`` and every non-ML path working, and using an ML
modeler fails with guidance pointing at ``pip install 'pulsar-core[ml]'``.
These checks run in subprocesses so the outer pytest process (which may
carry torch for test_ml.py) cannot mask a lazy-import leak.
"""

from __future__ import annotations

import subprocess
import sys

IMPORT_CHECK = """
import sys

import pulsar_core

assert "torch" not in sys.modules, "import pulsar_core must not load torch"
assert pulsar_core.MlpTorchScorer is not None
assert "mlp_torch" in pulsar_core.MODEL_REGISTRY.names()
assert "lstm_torch" in pulsar_core.MODEL_REGISTRY.names()
print("IMPORT-CLEAN")
"""

GUIDANCE_CHECK = """
import sys
from datetime import date

# simulate a torch-free install: block the torch import machinery
sys.modules["torch"] = None

import pulsar_core
from pulsar_core import CrossSection

scorer = pulsar_core.MODEL_REGISTRY.resolve("mlp_torch").create({}, ["f"])
section = CrossSection(as_of=date(2026, 1, 1), values={"f": {"A": 1.0}})
try:
    scorer.score(section, None)
except Exception as exc:
    message = str(exc)
    assert "pulsar-core[ml]" in message, message
    print("GUIDANCE-OK")
else:
    raise AssertionError("scoring without torch must fail with install guidance")
"""


def _run(script: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [sys.executable, "-c", script],
        capture_output=True,
        text=True,
        timeout=120,
    )


def test_import_pulsar_core_never_pulls_torch() -> None:
    completed = _run(IMPORT_CHECK)
    assert completed.returncode == 0, completed.stderr
    assert "IMPORT-CLEAN" in completed.stdout


def test_ml_paths_without_torch_fail_with_install_guidance() -> None:
    completed = _run(GUIDANCE_CHECK)
    assert completed.returncode == 0, completed.stderr
    assert "GUIDANCE-OK" in completed.stdout
