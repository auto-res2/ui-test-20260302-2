"""
Evaluation script for SCER-SC experiment.
Fetches results from WandB, generates comparison plots, and exports metrics.
"""

import os
import sys
import json
import argparse
from pathlib import Path
from typing import List, Dict, Any
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import wandb


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate SCER-SC experiment results")
    parser.add_argument(
        "--results_dir", type=str, required=True, help="Results directory"
    )
    parser.add_argument(
        "--run_ids", type=str, required=True, help="JSON string list of run IDs"
    )
    parser.add_argument(
        "--wandb_entity", type=str, default="airas", help="WandB entity"
    )
    parser.add_argument(
        "--wandb_project", type=str, default="ui-test-20260302-2", help="WandB project"
    )
    return parser.parse_args()


def fetch_run_from_wandb(entity: str, project: str, run_id: str) -> Dict[str, Any]:
    """
    Fetch run data from WandB by display name.

    Args:
        entity: WandB entity
        project: WandB project
        run_id: Run display name

    Returns:
        Dictionary with config, summary, and history
    """
    api = wandb.Api()

    # Find runs by display name
    runs = api.runs(
        f"{entity}/{project}", filters={"display_name": run_id}, order="-created_at"
    )

    if not runs:
        print(f"Warning: No run found with display name '{run_id}'")
        return None

    run = runs[0]  # Most recent run with that name

    # Fetch data
    config = run.config
    summary = run.summary._json_dict
    history = run.history()

    return {"config": config, "summary": summary, "history": history, "url": run.url}


def export_per_run_metrics(run_id: str, run_data: Dict[str, Any], results_dir: Path):
    """
    Export per-run metrics to JSON file.
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    # Extract key metrics
    metrics = {
        "run_id": run_id,
        "accuracy": run_data["summary"].get("accuracy", 0),
        "correct": run_data["summary"].get("correct", 0),
        "total": run_data["summary"].get("total", 0),
        "wandb_url": run_data["url"],
    }

    # Add method-specific info
    if "aggregation" in run_data["config"]:
        metrics["method"] = run_data["config"]["aggregation"].get("method", "unknown")

        if metrics["method"] == "scer_sc":
            metrics["lambda_L"] = run_data["summary"].get("lambda_L", None)
            metrics["lambda_E"] = run_data["summary"].get("lambda_E", None)
            metrics["lambda_C"] = run_data["summary"].get("lambda_C", None)
            metrics["lambda_H"] = run_data["summary"].get("lambda_H", None)

    # Save metrics
    with open(run_dir / "metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    print(f"Saved metrics for {run_id} to {run_dir / 'metrics.json'}")

    return metrics


def create_per_run_plots(run_id: str, run_data: Dict[str, Any], results_dir: Path):
    """
    Create per-run visualization plots.
    """
    run_dir = results_dir / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    history = run_data["history"]

    if len(history) == 0:
        print(f"Warning: No history data for {run_id}")
        return

    # Plot accuracy over time
    if "accuracy" in history.columns:
        plt.figure(figsize=(8, 5))
        plt.plot(history["processed"], history["accuracy"], marker="o", linewidth=2)
        plt.xlabel("Questions Processed")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy Progress - {run_id}")
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.savefig(run_dir / "accuracy_progress.pdf")
        plt.close()
        print(f"Saved accuracy plot for {run_id}")


def create_comparison_plots(all_metrics: List[Dict], results_dir: Path):
    """
    Create comparison plots across all runs.
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Set style
    sns.set_style("whitegrid")

    # Extract data
    run_ids = [m["run_id"] for m in all_metrics]
    accuracies = [m["accuracy"] for m in all_metrics]

    # Bar plot of accuracies
    plt.figure(figsize=(10, 6))
    colors = ["#FF6B6B" if "comparative" in rid else "#4ECDC4" for rid in run_ids]
    bars = plt.bar(
        range(len(run_ids)), accuracies, color=colors, edgecolor="black", linewidth=1.5
    )
    plt.xlabel("Method", fontsize=12)
    plt.ylabel("Accuracy", fontsize=12)
    plt.title("Accuracy Comparison Across Methods", fontsize=14, fontweight="bold")
    plt.xticks(range(len(run_ids)), run_ids, rotation=45, ha="right")
    plt.ylim(0, 1.0)
    plt.grid(axis="y", alpha=0.3)

    # Add value labels on bars
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + 0.01,
            f"{accuracies[i]:.3f}",
            ha="center",
            va="bottom",
            fontsize=10,
        )

    plt.tight_layout()
    plt.savefig(comparison_dir / "comparison_accuracy.pdf")
    plt.close()
    print(f"Saved comparison plot to {comparison_dir / 'comparison_accuracy.pdf'}")


def export_aggregated_metrics(all_metrics: List[Dict], results_dir: Path):
    """
    Export aggregated metrics across all runs.
    """
    comparison_dir = results_dir / "comparison"
    comparison_dir.mkdir(parents=True, exist_ok=True)

    # Identify proposed vs baseline runs
    proposed_runs = [m for m in all_metrics if m["run_id"].startswith("proposed")]
    baseline_runs = [m for m in all_metrics if m["run_id"].startswith("comparative")]

    # Find best in each category
    best_proposed = (
        max(proposed_runs, key=lambda x: x["accuracy"]) if proposed_runs else None
    )
    best_baseline = (
        max(baseline_runs, key=lambda x: x["accuracy"]) if baseline_runs else None
    )

    # Compute gap
    gap = None
    if best_proposed and best_baseline:
        gap = best_proposed["accuracy"] - best_baseline["accuracy"]

    # Create aggregated metrics
    aggregated = {
        "primary_metric": "accuracy",
        "metrics_by_run": {
            m["run_id"]: {"accuracy": m["accuracy"]} for m in all_metrics
        },
        "best_proposed": best_proposed["run_id"] if best_proposed else None,
        "best_proposed_accuracy": best_proposed["accuracy"] if best_proposed else None,
        "best_baseline": best_baseline["run_id"] if best_baseline else None,
        "best_baseline_accuracy": best_baseline["accuracy"] if best_baseline else None,
        "gap": gap,
    }

    # Save
    with open(comparison_dir / "aggregated_metrics.json", "w") as f:
        json.dump(aggregated, f, indent=2)

    print(f"Saved aggregated metrics to {comparison_dir / 'aggregated_metrics.json'}")
    print(f"\nSummary:")
    print(
        f"  Best Proposed: {best_proposed['run_id']} (accuracy={best_proposed['accuracy']:.4f})"
        if best_proposed
        else "  No proposed runs"
    )
    print(
        f"  Best Baseline: {best_baseline['run_id']} (accuracy={best_baseline['accuracy']:.4f})"
        if best_baseline
        else "  No baseline runs"
    )
    if gap is not None:
        print(f"  Gap: {gap:+.4f}")


def main():
    """Main evaluation function."""
    args = parse_args()

    # Parse run IDs
    run_ids = json.loads(args.run_ids)

    print("=" * 80)
    print("SCER-SC Experiment Evaluation")
    print("=" * 80)
    print(f"Results directory: {args.results_dir}")
    print(f"Run IDs: {run_ids}")
    print(f"WandB: {args.wandb_entity}/{args.wandb_project}")
    print("=" * 80)

    results_dir = Path(args.results_dir)
    results_dir.mkdir(parents=True, exist_ok=True)

    # Fetch data from WandB for each run
    all_metrics = []
    for run_id in run_ids:
        print(f"\nFetching data for {run_id}...")
        run_data = fetch_run_from_wandb(args.wandb_entity, args.wandb_project, run_id)

        if run_data is None:
            print(f"Skipping {run_id} (not found)")
            continue

        # Export per-run metrics
        metrics = export_per_run_metrics(run_id, run_data, results_dir)
        all_metrics.append(metrics)

        # Create per-run plots
        create_per_run_plots(run_id, run_data, results_dir)

    # Create comparison plots
    if all_metrics:
        print("\nCreating comparison plots...")
        create_comparison_plots(all_metrics, results_dir)

        # Export aggregated metrics
        export_aggregated_metrics(all_metrics, results_dir)

    print("\n" + "=" * 80)
    print("Evaluation completed successfully!")
    print("=" * 80)

    # Print all generated files
    print("\nGenerated files:")
    for run_id in run_ids:
        run_dir = results_dir / run_id
        if run_dir.exists():
            for file in run_dir.iterdir():
                print(f"  {file}")

    comparison_dir = results_dir / "comparison"
    if comparison_dir.exists():
        for file in comparison_dir.iterdir():
            print(f"  {file}")


if __name__ == "__main__":
    main()
