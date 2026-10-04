import os
import sys

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from eegrag.experiments.budget import plan_updates


def measurement(step_seconds=0.05, overhead_seconds=60):
    return {"seconds_per_step": step_seconds,
            "post_training_seconds_first_fold": overhead_seconds,
            "preprocessing_seconds_first_fold": 20,
            "encoder_setup_seconds_first_fold": 5, "load_seconds": 30,
            "n_test_folds": 23, "seeds": 1, "losses": 3,
            "epochs": 15, "configured_steps_per_epoch": 256}


def test_budget_keeps_requested_updates_when_all_stages_fit():
    plan = plan_updates(measurement())
    assert plan["steps_per_epoch"] == 256
    assert plan["runs"] == 69
    assert plan["training_hours"] == pytest.approx(3.68)
    assert plan["estimated_hours_with_margin"] + plan["reserved_hours"] <= 9.5


def test_budget_reduces_updates_using_measured_overhead_and_safety_margin():
    plan = plan_updates(measurement(step_seconds=0.15))
    assert 32 <= plan["steps_per_epoch"] < 256
    assert plan["estimated_hours_with_margin"] + plan["reserved_hours"] <= 9.5


def test_budget_rejects_sweep_when_evaluation_alone_exceeds_cap():
    with pytest.raises(ValueError, match="not launched"):
        plan_updates(measurement(overhead_seconds=600))


def test_budget_rejects_training_only_or_invalid_measurements():
    report = measurement()
    del report["post_training_seconds_first_fold"]
    with pytest.raises(ValueError, match="end-to-end"):
        plan_updates(report)
    report = measurement(float("nan"))
    with pytest.raises(ValueError, match="finite"):
        plan_updates(report)
