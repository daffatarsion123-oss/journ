"""Choose a declared update budget using an end-to-end timing measurement."""
import math


def plan_updates(report, max_hours=9.5, reserve_hours=1.0, safety_factor=1.35,
                 minimum_steps=32):
    required = ("seconds_per_step", "post_training_seconds_first_fold", "n_test_folds",
                "seeds", "losses", "epochs", "configured_steps_per_epoch")
    if any(key not in report for key in required):
        raise ValueError("Budget planning requires an end-to-end benchmark")
    if not all(math.isfinite(float(report[key])) and float(report[key]) > 0
               for key in required):
        raise ValueError("Benchmark values must be finite and positive")
    if not 0 <= reserve_hours < max_hours or safety_factor < 1 or minimum_steps < 1:
        raise ValueError("Invalid time budget or safety factor")
    runs = report["n_test_folds"] * report["seeds"] * report["losses"]
    per_run_overhead = (report.get("preprocessing_seconds_first_fold", 0)
                        + report.get("encoder_setup_seconds_first_fold", 0)
                        + report["post_training_seconds_first_fold"] + 15)
    overhead = per_run_overhead * runs + report.get("load_seconds", 0) * report["losses"]
    usable_seconds = (max_hours - reserve_hours) * 3600 / safety_factor - overhead
    steps = min(int(report["configured_steps_per_epoch"]), math.floor(
        usable_seconds / (runs * report["epochs"] * report["seconds_per_step"])))
    if steps < minimum_steps:
        raise ValueError("Measured overhead leaves too little training time; sweep not launched")
    training_seconds = runs * report["epochs"] * steps * report["seconds_per_step"]
    return {"steps_per_epoch": steps, "runs": runs,
            "training_hours": training_seconds / 3600,
            "overhead_hours": overhead / 3600,
            "estimated_hours_with_margin": (training_seconds + overhead) * safety_factor / 3600,
            "reserved_hours": reserve_hours, "max_hours": max_hours}
