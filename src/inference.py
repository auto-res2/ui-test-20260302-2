"""
Inference script for SCER-SC Chain-of-Thought experiment.
Handles LLM sampling, feature extraction, and answer aggregation.
"""

import os
import re
import json
import time
import random
from pathlib import Path
from typing import List, Dict, Any, Tuple
from collections import Counter
import numpy as np
from omegaconf import DictConfig, OmegaConf
import wandb
from tqdm import tqdm

import google.generativeai as genai

from src.preprocess import load_gsm8k, create_cot_prompt


def extract_final_answer(text: str) -> str:
    """
    Extract final answer from model output.
    Looks for "FINAL: <answer>" pattern, fallback to last number.
    """
    # Try to find FINAL: pattern
    final_match = re.search(r"FINAL:\s*([+-]?[\d,]+(?:\.\d+)?)", text, re.IGNORECASE)
    if final_match:
        # Remove commas from numbers
        return final_match.group(1).replace(",", "")

    # Fallback: find all numbers and return the last one
    numbers = re.findall(r"[+-]?[\d,]+(?:\.\d+)?", text)
    if numbers:
        return numbers[-1].replace(",", "")

    return ""


def compute_text_features(text: str) -> Dict[str, float]:
    """
    Compute text-local process-signal features for SCER-SC.

    Features:
    - len_i: word count (brevity prior)
    - edit_i: self-revision markers count
    - contra_i: contradiction markers count
    - hedge_i: uncertainty markers count
    """
    # Length: word count
    words = text.split()
    len_i = len(words)

    # Edit markers: self-revision patterns
    edit_patterns = [
        r"\bwait\b",
        r"\bactually\b",
        r"\blet me\b",
        r"\bcorrection\b",
        r"\bi made a mistake\b",
        r"\bno,?\s+that",
        r"\bhold on\b",
        r"\bsorry\b",
    ]
    edit_i = sum(
        len(re.findall(pattern, text, re.IGNORECASE)) for pattern in edit_patterns
    )

    # Contradiction markers
    contra_patterns = [
        r"\bnot\b.*\bbut\b",
        r"\binstead\b",
        r"\bhowever\b",
        r"\bon the other hand\b",
        r"\bcontrary\b",
    ]
    contra_i = sum(
        len(re.findall(pattern, text, re.IGNORECASE)) for pattern in contra_patterns
    )

    # Hedge markers: uncertainty language
    hedge_patterns = [
        r"\bmaybe\b",
        r"\bprobably\b",
        r"\bi think\b",
        r"\bseems\b",
        r"\bperhaps\b",
        r"\bpossibly\b",
        r"\bmight\b",
        r"\bcould be\b",
        r"\bunsure\b",
    ]
    hedge_i = sum(
        len(re.findall(pattern, text, re.IGNORECASE)) for pattern in hedge_patterns
    )

    return {
        "length": len_i,
        "edits": edit_i,
        "contradictions": contra_i,
        "hedges": hedge_i,
    }


def aggregate_answers(
    answers: List[str],
    features: List[Dict[str, float]],
    method: str,
    lambdas: Dict[str, float] | None = None,
) -> str:
    """
    Aggregate multiple sampled answers using specified method.

    Args:
        answers: List of extracted final answers
        features: List of text features for each answer
        method: Aggregation method (direct, vanilla_sc, scer_sc)
        lambdas: Lambda coefficients for SCER-SC weighting

    Returns:
        Final aggregated answer
    """
    if method == "direct":
        # Use only the first sample
        return answers[0] if answers else ""

    elif method == "vanilla_sc":
        # Majority vote
        answer_counts = Counter(answers)
        if not answer_counts:
            return ""
        return answer_counts.most_common(1)[0][0]

    elif method == "scer_sc":
        # Weighted vote with SCER-SC scoring
        if lambdas is None:
            lambdas = {
                "lambda_L": 0.001,
                "lambda_E": 0.1,
                "lambda_C": 0.2,
                "lambda_H": 0.05,
            }

        # Compute weights
        weights = []
        for feat in features:
            score = (
                lambdas["lambda_L"] * feat["length"]
                + lambdas["lambda_E"] * feat["edits"]
                + lambdas["lambda_C"] * feat["contradictions"]
                + lambdas["lambda_H"] * feat["hedges"]
            )
            weight = np.exp(-score)
            weights.append(weight)

        # Weighted vote
        answer_weights = {}
        for ans, w in zip(answers, weights):
            if ans not in answer_weights:
                answer_weights[ans] = 0
            answer_weights[ans] += w

        if not answer_weights:
            return ""

        return max(answer_weights.items(), key=lambda x: x[1])[0]

    else:
        raise ValueError(f"Unknown aggregation method: {method}")


def sample_from_llm(
    prompt: str, model, num_samples: int, temperature: float, max_tokens: int
) -> List[str]:
    """
    Sample multiple outputs from LLM.
    """
    outputs = []
    for _ in range(num_samples):
        try:
            response = model.generate_content(
                prompt,
                generation_config=genai.types.GenerationConfig(
                    temperature=temperature,
                    max_output_tokens=max_tokens,
                ),
            )
            outputs.append(response.text)
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error generating sample: {e}")
            outputs.append("")

    return outputs


def tune_lambdas_on_dev(
    dev_data: List[Dict], model, cfg: DictConfig, num_samples_sanity: int | None = None
) -> Dict[str, float]:
    """
    Simple grid search to tune lambda coefficients on dev set.
    """
    print("Tuning lambda coefficients on dev set...")

    # Use fewer samples in sanity_check mode
    k = (
        num_samples_sanity
        if num_samples_sanity is not None
        else cfg.inference.num_samples
    )

    # Grid search over lambda values
    lambda_L_values = [0.0001, 0.001, 0.01]
    lambda_E_values = [0.05, 0.1, 0.2]
    lambda_C_values = [0.1, 0.2, 0.5]
    lambda_H_values = [0.01, 0.05, 0.1]

    best_accuracy = 0
    best_lambdas = {
        "lambda_L": cfg.aggregation.lambda_L,
        "lambda_E": cfg.aggregation.lambda_E,
        "lambda_C": cfg.aggregation.lambda_C,
        "lambda_H": cfg.aggregation.lambda_H,
    }

    # Sample outputs for dev set (reuse for all lambda configs)
    print(f"Generating {k} samples for {len(dev_data)} dev examples...")
    dev_samples = []
    for example in tqdm(dev_data, desc="Dev sampling"):
        prompt = create_cot_prompt(example["question"])
        outputs = sample_from_llm(
            prompt, model, k, cfg.model.temperature, cfg.model.max_tokens
        )

        answers = [extract_final_answer(out) for out in outputs]
        features = [compute_text_features(out) for out in outputs]

        dev_samples.append(
            {
                "ground_truth": example["answer"],
                "answers": answers,
                "features": features,
            }
        )

    # Grid search
    print("Running grid search...")
    for lambda_L in lambda_L_values:
        for lambda_E in lambda_E_values:
            for lambda_C in lambda_C_values:
                for lambda_H in lambda_H_values:
                    lambdas = {
                        "lambda_L": lambda_L,
                        "lambda_E": lambda_E,
                        "lambda_C": lambda_C,
                        "lambda_H": lambda_H,
                    }

                    # Evaluate on dev set
                    correct = 0
                    for sample in dev_samples:
                        pred = aggregate_answers(
                            sample["answers"], sample["features"], "scer_sc", lambdas
                        )
                        if pred.strip() == sample["ground_truth"].strip():
                            correct += 1

                    accuracy = correct / len(dev_samples) if dev_samples else 0

                    if accuracy > best_accuracy:
                        best_accuracy = accuracy
                        best_lambdas = lambdas.copy()

    print(f"Best lambdas: {best_lambdas}, dev accuracy: {best_accuracy:.4f}")
    return best_lambdas


def run_inference(cfg: DictConfig):
    """
    Main inference function.
    """
    # Initialize WandB
    if cfg.wandb.mode != "disabled":
        # In sanity_check mode, use separate project namespace
        project = cfg.wandb.project
        if cfg.mode == "sanity_check":
            project = f"{cfg.wandb.project}-sanity"

        wandb.init(
            entity=cfg.wandb.entity,
            project=project,
            name=cfg.run.run_id,
            config=OmegaConf.to_container(cfg, resolve=True),
            mode=cfg.wandb.mode,
        )
        print(f"WandB run: {wandb.run.url}")

    # Initialize Gemini API
    api_key = os.environ.get("GOOGLE_API_KEY")
    if not api_key:
        raise ValueError("GOOGLE_API_KEY environment variable not set")

    genai.configure(api_key=api_key)
    model = genai.GenerativeModel(cfg.model.name)

    # Load dataset
    print(f"Loading GSM8K dataset...")
    data_splits = load_gsm8k(
        cache_dir=cfg.cache_dir,
        subset_size=cfg.dataset.subset_size,
        dev_size=cfg.dataset.dev_size,
        seed=cfg.inference.seed,
    )

    dev_data = data_splits["dev"]
    test_data = data_splits["test"]

    # Adjust for sanity_check mode
    num_samples = cfg.inference.num_samples
    if cfg.mode == "sanity_check":
        # Reduce to 5 samples for sanity check
        num_samples = 5
        test_data = test_data[:10]  # Use only 10 test examples
        dev_data = dev_data[:5]  # Use only 5 dev examples

    # Tune lambdas on dev set (only for SCER-SC)
    lambdas = None
    if cfg.aggregation.method == "scer_sc" and cfg.aggregation.get(
        "tune_on_dev", False
    ):
        lambdas = tune_lambdas_on_dev(
            dev_data, model, cfg, num_samples if cfg.mode == "sanity_check" else None
        )
    else:
        lambdas = {
            "lambda_L": cfg.aggregation.get("lambda_L", 0.001),
            "lambda_E": cfg.aggregation.get("lambda_E", 0.1),
            "lambda_C": cfg.aggregation.get("lambda_C", 0.2),
            "lambda_H": cfg.aggregation.get("lambda_H", 0.05),
        }

    # Run inference on test set
    print(f"Running inference on {len(test_data)} test examples...")
    results = []
    correct = 0
    total_samples_generated = 0

    for i, example in enumerate(tqdm(test_data, desc="Test inference")):
        prompt = create_cot_prompt(example["question"])
        outputs = sample_from_llm(
            prompt, model, num_samples, cfg.model.temperature, cfg.model.max_tokens
        )

        answers = [extract_final_answer(out) for out in outputs]
        features = [compute_text_features(out) for out in outputs]

        # Aggregate
        final_answer = aggregate_answers(
            answers, features, cfg.aggregation.method, lambdas
        )

        is_correct = final_answer.strip() == example["answer"].strip()
        if is_correct:
            correct += 1

        total_samples_generated += len(outputs)

        results.append(
            {
                "question": example["question"],
                "ground_truth": example["answer"],
                "prediction": final_answer,
                "correct": is_correct,
                "all_answers": answers,
                "all_features": features,
            }
        )

        # Log intermediate metrics
        if cfg.wandb.mode != "disabled" and (i + 1) % 10 == 0:
            wandb.log({"processed": i + 1, "accuracy": correct / (i + 1)})

    # Compute final metrics
    accuracy = correct / len(test_data) if test_data else 0

    print(f"\nFinal Results:")
    print(f"  Accuracy: {accuracy:.4f} ({correct}/{len(test_data)})")
    print(f"  Total samples generated: {total_samples_generated}")

    # Save results
    results_dir = Path(cfg.results_dir) / cfg.run.run_id
    results_dir.mkdir(parents=True, exist_ok=True)

    with open(results_dir / "results.json", "w") as f:
        json.dump(results, f, indent=2)

    with open(results_dir / "metrics.json", "w") as f:
        json.dump(
            {
                "accuracy": accuracy,
                "correct": correct,
                "total": len(test_data),
                "num_samples_per_question": num_samples,
                "method": cfg.aggregation.method,
                "lambdas": lambdas if cfg.aggregation.method == "scer_sc" else None,
            },
            f,
            indent=2,
        )

    # Log to WandB
    if cfg.wandb.mode != "disabled":
        wandb.log(
            {"final_accuracy": accuracy, "correct": correct, "total": len(test_data)}
        )
        wandb.summary["accuracy"] = accuracy
        wandb.summary["correct"] = correct
        wandb.summary["total"] = len(test_data)

        if cfg.aggregation.method == "scer_sc":
            wandb.summary["lambda_L"] = lambdas["lambda_L"]
            wandb.summary["lambda_E"] = lambdas["lambda_E"]
            wandb.summary["lambda_C"] = lambdas["lambda_C"]
            wandb.summary["lambda_H"] = lambdas["lambda_H"]

    # Sanity validation
    if cfg.mode == "sanity_check":
        perform_sanity_validation(total_samples_generated, accuracy)

    # Finish WandB
    if cfg.wandb.mode != "disabled":
        wandb.finish()

    print(f"\nResults saved to {results_dir}")


def perform_sanity_validation(samples_processed: int, accuracy: float):
    """
    Perform sanity validation checks for inference task.
    """
    # Check: at least 5 samples processed
    samples_ok = samples_processed >= 5

    # Check: accuracy is finite and reasonable
    accuracy_ok = np.isfinite(accuracy) and 0 <= accuracy <= 1

    # Check: not completely failing (accuracy > 0)
    not_zero = accuracy > 0

    all_ok = samples_ok and accuracy_ok and not_zero

    # Print summary
    summary = {
        "samples": samples_processed,
        "accuracy": accuracy,
        "status": "pass" if all_ok else "fail",
    }
    print(f"SANITY_VALIDATION_SUMMARY: {json.dumps(summary)}")

    # Print verdict
    if all_ok:
        print("SANITY_VALIDATION: PASS")
    else:
        reasons = []
        if not samples_ok:
            reasons.append("insufficient_samples")
        if not accuracy_ok:
            reasons.append("invalid_accuracy")
        if not not_zero:
            reasons.append("zero_accuracy")
        print(f"SANITY_VALIDATION: FAIL reason={','.join(reasons)}")
