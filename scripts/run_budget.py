"""Benchmark, plan and run the three LOSO losses within a wall-clock cap."""
from _bootstrap import base_parser, load
import copy
import json
import os
import subprocess
import sys
import time


def main():
    parser = base_parser(__doc__)
    parser.set_defaults(config="configs/contrastive_runpod_budget.yaml")
    parser.add_argument("--max-hours", type=float, default=9.5)
    parser.add_argument("--hourly-rate", type=float)
    parser.add_argument("--plan-only", action="store_true")
    args = parser.parse_args()
    if not 1 < args.max_hours <= 10:
        parser.error("--max-hours must be >1 and <=10")
    cfg = load(args)
    from eegrag.config import dump_config
    from eegrag.experiments.budget import plan_updates
    from eegrag.utils.io import ensure_dir, save_json
    if cfg.training.device != "cuda" or not cfg.training.tensor_batches:
        parser.error("Budget runner requires CUDA tensor batches")
    started = time.monotonic()
    deadline = started + args.max_hours * 3600
    folder = ensure_dir(os.path.join(cfg.output_dir, "budget_plan"))
    config_path = os.path.join(folder, "benchmark_config.yaml")
    dump_config(cfg, config_path)
    def run(command):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise subprocess.TimeoutExpired(command, args.max_hours * 3600)
        subprocess.run(command, check=True, timeout=remaining)
    try:
        run([sys.executable, "scripts/benchmark_runpod.py", "--config", config_path,
             "--steps", "30", "--losses", "3", "--end-to-end"])
        with open(os.path.join(cfg.output_dir, "runpod_benchmark.json")) as fh:
            report = json.load(fh)
        if report["gpu"] == "cpu":
            raise ValueError("CPU timing cannot authorize a GPU budget")
        plan = plan_updates(report, max_hours=(deadline - time.monotonic()) / 3600)
        plan["benchmark_elapsed_hours"] = (time.monotonic() - started) / 3600
        if args.hourly_rate is not None:
            plan["maximum_compute_cost"] = args.hourly_rate * args.max_hours
        save_json(plan, os.path.join(folder, "plan.json"))
        print("\nDeclared budget plan:\n" + json.dumps(plan, indent=2), flush=True)
        if args.plan_only:
            return
        for loss in ("supcon", "balanced_supcon", "imbalance_supcon"):
            run_cfg = copy.deepcopy(cfg)
            run_cfg.loss.name = loss
            run_cfg.training.steps_per_epoch = plan["steps_per_epoch"]
            run_cfg.name = f"runpod_budget_{loss}_steps{plan['steps_per_epoch']}"
            path = os.path.join(folder, f"{loss}.yaml")
            dump_config(run_cfg, path)
            run([sys.executable, "scripts/run_loso_contrastive.py", "--config", path])
    except subprocess.TimeoutExpired:
        raise SystemExit("Wall-clock cap reached. Training stopped; completed epochs remain checkpointed. Stop the pod to end billing.")
    except ValueError as exc:
        raise SystemExit(str(exc))
    print("Budget sweep complete. Stop the pod to end billing.")


if __name__ == "__main__":
    main()
