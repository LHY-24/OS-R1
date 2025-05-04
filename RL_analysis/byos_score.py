# Gemini enhanced version of the BYOS scoring function byos_score.py
# Key Changes Incorporated:

# Clearer Parsing: parse_config_string uses regex for better handling of =.
# Standard Metrics: calculate_content_metrics computes precision, recall, F1 (for names), and value accuracy (for matching names).
# Weighted Content Score: compute_score_content_byos combines F1 and Value Accuracy with defined weights (WEIGHT_NAME_F1, WEIGHT_VALUE_ACCURACY).
# Relaxed Format Score: compute_score_format_byos primarily checks for <answer> presence and structure, giving partial credit and applying specific penalties.
# Weighted Final Reward: compute_final_reward_byos combines format and content using weights (WEIGHT_FORMAT_SCORE, WEIGHT_CONTENT_SCORE) without gating. Starts from 0 and applies penalties (garbage, missing answer) or bonuses.
# Debug Prints: Includes print statements to trace calculations (can be commented out later).
# Robustness: Added try-except blocks and handling for empty/invalid inputs.
# Example Usage: Includes a if __name__ == '__main__': block demonstrating how the functions work with various inputs.

# --- START OF FILE byos_score.py ---

import re
import json
from typing import Dict, Set, Tuple, Optional, List, Any

# --- Configuration Constants ---
# Weights for combining content metrics into a single score
# Prioritize getting the right configs (Recall/F1) and their values correct
WEIGHT_NAME_F1 = 0.5
WEIGHT_VALUE_ACCURACY = 0.5

# Weights for combining format and content scores into the final reward
WEIGHT_FORMAT_SCORE = 0.3
WEIGHT_CONTENT_SCORE = 0.7

# Penalties
PENALTY_JSON_ERROR = -0.5
PENALTY_MISSING_ANSWER = -0.3
PENALTY_EMPTY_ANSWER = -0.1
PENALTY_GARBAGE_OUTPUT = -0.7 # If output is clearly nonsensical
PENALTY_UNEXPECTED_ERROR = -0.4

# Bonuses
BONUS_BASE_SUCCESS = 0.0 # Start from 0, add positive contributions
BONUS_FORMAT_BASE = 0.2 # Basic structure like <|im_end|>
BONUS_FORMAT_ANSWER_PRESENT = 0.6
BONUS_FORMAT_THINK_PRESENT = 0.2 # Optional bonus


# --- Helper Functions ---

def parse_config_string(config_str: Optional[str]) -> Tuple[Set[str], Dict[str, str]]:
    """
    Parses a multi-line kernel configuration string into a set of config names
    and a dictionary mapping names to their string values. Handles comments
    and common formats like NAME=y, NAME=n, NAME=123, NAME="value".

    Args:
        config_str: The raw configuration string, potentially multi-line.

    Returns:
        A tuple containing:
        - config_names (Set[str]): A set of unique CONFIG_ names found.
        - config_values (Dict[str, str]): A dictionary mapping config names to values.
    """
    config_names = set()
    config_values = {}
    if not isinstance(config_str, str):
        print("[Parse WARN] Input config_str is not a string.")
        return config_names, config_values

    lines = config_str.strip().split('\n')
    for i, line in enumerate(lines):
        line = line.strip()
        if not line or line.startswith('#'): # Skip empty lines and comments
            continue

        # Regex to capture NAME=value, handling potential whitespace and quotes
        # Captures CONFIG_NAME, then =, then the value (non-greedy)
        match = re.match(r'^(CONFIG_[A-Za-z0-9_]+)\s*=\s*(.*)$', line)
        if match:
            name, value = match.groups()
            # Strip potential quotes from value, keep original case for value comparison (e.g., y vs n)
            value = value.strip().strip('"').strip("'")
            config_names.add(name)
            config_values[name] = value
            # print(f"[Parse DEBUG] Parsed Line {i+1}: '{name}' = '{value}'") # Debug
        else:
            print(f"[Parse WARN] Skipping malformed config line {i+1}: '{line}'")

    return config_names, config_values

def calculate_content_metrics(solution_names: Set[str], truth_names: Set[str],
                              solution_vals: Dict[str, str], truth_vals: Dict[str, str]) -> Dict[str, float]:
    """
    Calculates precision, recall, F1 for config names, and value accuracy
    for the configs present in both solution and ground truth.

    Args:
        solution_names: Set of config names extracted from the agent's answer.
        truth_names: Set of config names extracted from the ground truth.
        solution_vals: Dict mapping solution config names to values.
        truth_vals: Dict mapping ground truth config names to values.

    Returns:
        A dictionary containing calculated metrics.
    """
    metrics = {
        "precision": 0.0, "recall": 0.0, "f1": 0.0,
        "value_accuracy": 0.0, "num_correct_values": 0.0,
        "num_intersection": 0.0, "num_solution": float(len(solution_names)),
        "num_truth": float(len(truth_names))
    }

    if not truth_names: # Avoid division by zero if ground truth is empty
        print("[Metrics WARN] Ground truth config set is empty.")
        return metrics # Return zeros

    intersection_names = solution_names.intersection(truth_names)
    metrics["num_intersection"] = float(len(intersection_names))

    # Name Precision, Recall, F1
    if len(solution_names) > 0:
        metrics["precision"] = len(intersection_names) / len(solution_names)
    if len(truth_names) > 0: # Already checked truth_names is not empty
        metrics["recall"] = len(intersection_names) / len(truth_names)
    if (metrics["precision"] + metrics["recall"]) > 0:
        metrics["f1"] = 2 * (metrics["precision"] * metrics["recall"]) / (metrics["precision"] + metrics["recall"])

    # Value Accuracy (only for configs present in both)
    correct_values_count = 0
    if len(intersection_names) > 0:
        for name in intersection_names:
            sol_val = solution_vals.get(name)
            truth_val = truth_vals.get(name)
            # Use case-sensitive comparison for values like 'y', 'n'
            if sol_val == truth_val:
                correct_values_count += 1
        metrics["value_accuracy"] = correct_values_count / len(intersection_names)
        metrics["num_correct_values"] = float(correct_values_count)
    else:
         # If no common configs, value accuracy is trivially perfect for that empty set.
         # Depending on interpretation, 0.0 might also be valid. Let's use 1.0.
         metrics["value_accuracy"] = 1.0

    return metrics

def extract_solution(agent_output: Optional[str]) -> Optional[str]:
    """
    Extracts the content from the first occurrence of <answer>...</answer>.

    Args:
        agent_output: The full string output from the agent assistant turn.

    Returns:
        The extracted content as a string, or None if the tag is not found
        or the input is invalid.
    """
    if not isinstance(agent_output, str):
        return None
    # Use non-greedy match .*? to capture content between the first pair
    answer_pattern = r'<answer>(.*?)</answer>'
    match = re.search(answer_pattern, agent_output, re.DOTALL | re.IGNORECASE) # Make tag matching case-insensitive
    return match.group(1).strip() if match else None

# --- Main Reward Calculation Functions ---

def compute_score_format_byos(agent_output: str) -> float:
    """
    Calculates a format score based on structure, focusing on <answer> presence.

    Args:
        agent_output: The full string output from the agent assistant turn.

    Returns:
        A format score between 0.0 and 1.0.
    """
    if not isinstance(agent_output, str):
        return 0.0 # Invalid input

    format_score = 0.0

    # Basic check for ending tag
    if agent_output.strip().endswith("<|im_end|>"):
        format_score += BONUS_FORMAT_BASE # e.g., 0.2

    # Primary check: Presence and extractability of <answer>
    answer_content = extract_solution(agent_output)
    if answer_content is not None:
        format_score += BONUS_FORMAT_ANSWER_PRESENT # e.g., 0.6
        if not answer_content: # Check if the extracted content is empty
             format_score += PENALTY_EMPTY_ANSWER # e.g., -0.1 (net +0.5 if present but empty)
    else:
        # Significant penalty if <answer> block is missing entirely
        format_score += PENALTY_MISSING_ANSWER # e.g., -0.3

    # Optional: Check for <think> preceding the *final* answer block
    # This regex looks for think, then answer, then the end tag, ignoring whitespace
    think_answer_pattern = r'<think>(.*?)<\/think>\s*<answer>.*?<\/answer>\s*<\|im_end\|>\s*$'
    if re.search(think_answer_pattern, agent_output.strip(), re.DOTALL | re.IGNORECASE):
         format_score += BONUS_FORMAT_THINK_PRESENT # e.g., 0.2

    # Ensure score is clipped between 0.0 and 1.0
    return max(0.0, min(format_score, 1.0))

def compute_score_content_byos(answer_content: Optional[str], ground_truth: str) -> Tuple[float, Dict[str, float]]:
    """
    Computes a content score based on Name F1 and Value Accuracy.

    Args:
        answer_content: The string extracted from the <answer> tag.
        ground_truth: The ground truth configuration string.

    Returns:
        A tuple containing:
        - content_score: The combined content score (0.0 to 1.0).
        - metrics: A dictionary with detailed precision, recall, f1, value_accuracy, etc.
    """
    detailed_metrics = {}
    if answer_content is None:
        print("[Content Score WARN] No answer content provided.")
        return 0.0, detailed_metrics
    if not isinstance(ground_truth, str) or not ground_truth.strip():
        print("[Content Score WARN] Ground truth is missing or empty.")
        return 0.0, detailed_metrics # Cannot score without ground truth

    try:
        solution_names, solution_vals = parse_config_string(answer_content)
        truth_names, truth_vals = parse_config_string(ground_truth)

        # Handle case where parsing fails or returns empty sets
        if not truth_names:
             print("[Content Score WARN] Ground truth parsing resulted in empty set.")
             return 0.0, {"num_truth": 0.0} # Return zero score if ground truth is effectively empty

        metrics = calculate_content_metrics(solution_names, truth_names, solution_vals, truth_vals)
        detailed_metrics.update(metrics) # Store detailed metrics

        # Combine metrics into a single score (weighted sum)
        content_score = (WEIGHT_NAME_F1 * metrics["f1"] +
                         WEIGHT_VALUE_ACCURACY * metrics["value_accuracy"])

        # Ensure score is bounded [0, 1]
        content_score = max(0.0, min(content_score, 1.0))

        return content_score, detailed_metrics

    except Exception as e:
        print(f"[Content Score ERROR] Unexpected error during content scoring: {e}")
        import traceback
        traceback.print_exc()
        return 0.0, {"error": str(e)} # Return 0 score on error

def compute_final_reward_byos(agent_output: str, ground_truth: str) -> Tuple[float, Dict[str, float]]:
    """
    Calculates the final scalar reward for the BYOS task for use in RL.

    Combines format and content scores using predefined weights. Includes basic
    penalties for major failures.

    Args:
        agent_output: The full string output from the agent assistant turn.
        ground_truth: The target kernel configuration string.

    Returns:
        A tuple containing:
        - final_reward: The scalar reward value (clipped between -1.0 and 1.0).
        - metrics: A dictionary containing detailed sub-scores and the final reward.
    """
    metrics = {}
    final_reward = BONUS_BASE_SUCCESS # Start from 0

    try:
        # 1. Basic Output Check (Garbage Penalty)
        if not agent_output or (isinstance(agent_output, str) and agent_output.strip().startswith("!!!!!!!!")):
            print("[Reward Final] Applying penalty for garbage/empty output.")
            final_reward += PENALTY_GARBAGE_OUTPUT # e.g., -0.7
            metrics["reward/penalty_garbage"] = PENALTY_GARBAGE_OUTPUT
            # Clip immediately if garbage detected, no further scoring needed
            final_reward = max(-1.0, min(final_reward, 1.0))
            metrics["reward/final_reward"] = final_reward
            return final_reward, metrics

        # 2. Calculate Format Score
        format_score = compute_score_format_byos(agent_output)
        metrics["reward/format_score"] = format_score

        # 3. Extract Answer Content
        answer_content = extract_solution(agent_output)

        # 4. Calculate Content Score (only if answer is extractable)
        content_score = 0.0
        content_metrics = {}
        if answer_content is not None:
            content_score, content_metrics = compute_score_content_byos(answer_content, ground_truth)
        else:
             # Apply penalty if format score was high but answer extraction failed
             if format_score > 0.5: # Threshold indicates likely presence of tags
                 print("[Reward Final] Penalizing: Format seemed OK, but <answer> content extraction failed.")
                 final_reward += PENALTY_MISSING_ANSWER # Apply penalty for missing/unextractable answer
             else:
                 print("[Reward Final] No <answer> content found or extracted.")


        metrics["reward/content_score"] = content_score
        metrics.update({f"content/{k}": v for k, v in content_metrics.items()})

        # 5. Combine Scores (Weighted Sum)
        # Apply format and content contributions to the base reward (which started at 0)
        final_reward += (WEIGHT_FORMAT_SCORE * format_score) + (WEIGHT_CONTENT_SCORE * content_score)

        # 6. Final Clipping
        final_reward = max(-1.0, min(final_reward, 1.0))
        metrics["reward/final_reward"] = final_reward

        print(f"[Reward Final] Format={format_score:.2f}, Content={content_score:.2f} => Final Reward={final_reward:.3f}")
        # print(f"[Reward Metrics] {metrics}") # Uncomment for very detailed logging

        return final_reward, metrics

    except Exception as e:
        print(f"[Reward Final ERROR] Unexpected error in compute_final_reward_byos: {e}")
        import traceback
        traceback.print_exc()
        metrics["reward/error"] = 1.0
        metrics["reward/final_reward"] = PENALTY_UNEXPECTED_ERROR # Apply a generic penalty
        return PENALTY_UNEXPECTED_ERROR, metrics

# --- Example Usage (for testing) ---
if __name__ == '__main__':
    print("--- Testing BYOS Score Functions ---")

    # Example Ground Truth
    gt = """
# Basic Configs
CONFIG_X86_64=y
CONFIG_SMP=y
# Performance Related
CONFIG_HZ_1000=y
CONFIG_PREEMPT_VOLUNTARY=y
# Filesystem
CONFIG_EXT4_FS=y
CONFIG_FS_ENCRYPTION=n
# Debugging (should ideally not be in agent output unless requested)
CONFIG_DEBUG_INFO=n
    """

    # Example Agent Outputs
    test_cases = [
        {
            "name": "Perfect Match",
            "output": """<|im_start|>assistant
<think>The user wants a performant config. I should enable SMP, high HZ, and voluntary preempt. EXT4 is standard.</think>
<answer>
CONFIG_X86_64=y
CONFIG_SMP=y
CONFIG_HZ_1000=y
CONFIG_PREEMPT_VOLUNTARY=y
CONFIG_EXT4_FS=y
CONFIG_FS_ENCRYPTION=n
CONFIG_DEBUG_INFO=n
</answer><|im_end|>""",
            "gt": gt
        },
        {
            "name": "Missing Configs, Correct Values",
            "output": """<|im_start|>assistant
<think>Focus on core performance.</think>
<answer>
CONFIG_X86_64=y
CONFIG_SMP=y
CONFIG_HZ_1000=y
CONFIG_PREEMPT_VOLUNTARY=y
</answer><|im_end|>""",
            "gt": gt
        },
        {
            "name": "Extra Configs, Correct Values",
            "output": """<|im_start|>assistant
<think>Adding network support just in case.</think>
<answer>
CONFIG_X86_64=y
CONFIG_SMP=y
CONFIG_HZ_1000=y
CONFIG_PREEMPT_VOLUNTARY=y
CONFIG_EXT4_FS=y
CONFIG_FS_ENCRYPTION=n
CONFIG_DEBUG_INFO=n
CONFIG_NET_CORE=y
CONFIG_TCP_CONG_CUBIC=y
</answer><|im_end|>""",
            "gt": gt
        },
        {
             "name": "Wrong Value",
            "output": """<|im_start|>assistant
<think>Maybe encryption helps?</think>
<answer>
CONFIG_X86_64=y
CONFIG_SMP=y
CONFIG_HZ_1000=y
CONFIG_PREEMPT_VOLUNTARY=y
CONFIG_EXT4_FS=y
CONFIG_FS_ENCRYPTION=y
CONFIG_DEBUG_INFO=n
</answer><|im_end|>""",
            "gt": gt
        },
        {
            "name": "Malformed Answer Content",
            "output": """<|im_start|>assistant
<think>Oops, format error.</think>
<answer>
CONFIG_X86_64 y
CONFIG_SMP=y=y
HZ_1000=y
</answer><|im_end|>""",
            "gt": gt
        },
         {
            "name": "Missing Answer Tag",
            "output": """<|im_start|>assistant
<think>I'll just list them.</think>
CONFIG_X86_64=y
CONFIG_SMP=y
<|im_end|>""",
            "gt": gt
        },
        {
            "name": "Garbage Output",
            "output": """!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!<|im_end|>""",
            "gt": gt
        },
         {
            "name": "Empty Answer Tag",
            "output": """<|im_start|>assistant
<think>I am unsure.</think>
<answer></answer><|im_end|>""",
            "gt": gt
        }
    ]

    for case in test_cases:
        print(f"\n--- Running Test Case: {case['name']} ---")
        final_reward, metrics = compute_final_reward_byos(case['output'], case['gt'])
        print(f"Output:\n{case['output'][:200]}...")
        print(f"Ground Truth:\n{case['gt'][:200]}...")
        print(f"Final Reward: {final_reward:.4f}")
        print("Detailed Metrics:")
        for k, v in metrics.items():
            print(f"  {k}: {v:.4f}")

# --- END OF FILE byos_score.py ---

