"""
Main orchestrator for SCER-SC experiment.
Handles configuration loading and invokes inference script.
"""

import sys
import hydra
from omegaconf import DictConfig, OmegaConf

from src.inference import run_inference


@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    """
    Main entry point for SCER-SC experiment.

    This orchestrator loads the Hydra config and invokes the appropriate script.
    For this experiment (inference-only), it calls run_inference directly.
    """
    print("=" * 80)
    print("SCER-SC Chain-of-Thought Experiment")
    print("=" * 80)
    print(f"\nRun ID: {cfg.run.run_id}")
    print(f"Method: {cfg.run.method.name} ({cfg.run.method.type})")
    print(f"Model: {cfg.run.model.name}")
    print(f"Dataset: {cfg.run.dataset.name}")
    print(f"Mode: {cfg.mode}")
    print(f"Results dir: {cfg.results_dir}")
    print("\nConfig:")
    print(OmegaConf.to_yaml(cfg))
    print("=" * 80)

    # Apply mode-specific overrides
    if cfg.mode == "sanity_check":
        print("\nApplying sanity_check mode overrides...")
        # Already handled in inference.py

    # Run inference
    try:
        run_inference(cfg)
        print("\n" + "=" * 80)
        print("Experiment completed successfully!")
        print("=" * 80)
    except Exception as e:
        print(f"\nError during experiment: {e}", file=sys.stderr)
        import traceback

        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
