"""
Dataset loading and preprocessing for SCER-SC experiment.
"""

import os
from datasets import load_dataset
from typing import Dict, List, Any


def load_gsm8k(
    cache_dir: str, subset_size: int = 200, dev_size: int = 50, seed: int = 42
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Load GSM8K dataset and split into dev and test sets.

    Args:
        cache_dir: Directory to cache the dataset
        subset_size: Total number of examples to use
        dev_size: Number of examples for dev set (for lambda tuning)
        seed: Random seed for reproducibility

    Returns:
        Dictionary with 'dev' and 'test' splits
    """
    # Load GSM8K test split
    dataset = load_dataset("gsm8k", "main", split="test", cache_dir=cache_dir)

    # Shuffle and select subset
    dataset = dataset.shuffle(seed=seed).select(range(subset_size))

    # Split into dev and test
    dev_data = dataset.select(range(dev_size))
    test_data = dataset.select(range(dev_size, subset_size))

    # Convert to list of dicts for easier processing
    dev_examples = []
    for item in dev_data:
        # Extract numeric answer from answer string (format: "#### 123")
        answer_str = item["answer"].split("####")[-1].strip()
        dev_examples.append(
            {
                "question": item["question"],
                "answer": answer_str,
                "full_solution": item["answer"],
            }
        )

    test_examples = []
    for item in test_data:
        answer_str = item["answer"].split("####")[-1].strip()
        test_examples.append(
            {
                "question": item["question"],
                "answer": answer_str,
                "full_solution": item["answer"],
            }
        )

    return {"dev": dev_examples, "test": test_examples}


def create_cot_prompt(question: str) -> str:
    """
    Create a Chain-of-Thought prompt for a given question.

    Args:
        question: The math question to solve

    Returns:
        Formatted prompt string
    """
    prompt = f"""Solve the following math problem step by step. Show your reasoning clearly.

Question: {question}

Please provide:
1. Your step-by-step reasoning
2. Your final answer on a new line starting with "FINAL: " followed by just the numeric answer

Let's solve this step by step:"""

    return prompt
