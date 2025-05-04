### V2: Enhanced version of os_knowledge_tool.py

### Key Changes include:
# Static Configuration: Tool parameters like working_dir, search_mode, llm_model are set during initialization via constructor arguments or environment variables.
# Lazy Initialization: The LightRAG core is initialized only on the first execute call.
# Robust Error Handling: Clear distinction between initialization errors and execution errors, returning structured JSON error messages. Includes directory checks.
# Accurate Parameter Schema & Validation: The description clarifies conditional argument requirements, and execute performs runtime validation.
# Refined Reward Logic: calculate_reward focuses on execution success, basic content quality heuristics (length), and optional ground-truth similarity, removing the flawed checks from the previous version.
# Mocking: Includes MockLightRAG for testing without the full dependency.
# Clearer Logging: Added informative print statements (can be commented out in production).
# Type Hinting & Docstrings: Improved documentation.

# --- START OF FILE os_knowledge_tool.py ---

import os
import json
from typing import Dict, List, Any, Optional, Tuple, Type

# --- Tool Base Import ---
try:
    # Assuming agent_r1 package structure for imports
    from agent_r1.tool.tool_base import Tool
    print("[INFO] Imported Tool base class from agent_r1.")
except ImportError:
    # Fallback for standalone execution or different structure
    print("[WARN] Could not import Tool from agent_r1.tool.tool_base. Using placeholder.")
    class Tool: # Basic placeholder for Tool base class
        def __init__(self, name: str, description: str, parameters: Dict):
            self.name = name
            self.description = description
            self.parameters = parameters
        def execute(self, args: Dict) -> str: raise NotImplementedError("Placeholder execute")
        def calculate_reward(self, args: Dict, result: str) -> float: return 0.0
        def batch_execute(self, args_list: List[Dict]) -> List[str]: return [self.execute(args) for args in args_list]
        def get_description(self) -> Dict: return {"type": "function", "function": {"name": self.name, "description": self.description, "parameters": self.parameters}}

# --- LightRAG Imports & Mocking ---
# Attempt to import real LightRAG components, fallback to Mock if unavailable
try:
    from lightrag import LightRAG, QueryParam
    from lightrag.llm import gpt_4o_mini_complete, gpt_4o_complete
    LightRAGComponent: Type = LightRAG # Use the real LightRAG class
    print("[INFO] Using real LightRAG components.")

except ImportError:
    print("[WARN] Real LightRAG library not found. Using MockLightRAG for simulation.")
    print("[WARN] Tool will simulate RAG responses and LLM calls.")

    # Define placeholder QueryParam if real one isn't available
    class QueryParam:
        def __init__(self, mode: str = "hybrid", **kwargs):
            self.mode = mode
            self.kwargs = kwargs
        def __repr__(self):
            return f"QueryParam(mode='{self.mode}', kwargs={self.kwargs})"

    # Define mock LLM functions
    def gpt_4o_mini_complete(**kwargs) -> str:
        prompt = kwargs.get('prompt', '')
        return f"Mocked GPT-4o-mini completion for: '{prompt[:70]}...'"
    def gpt_4o_complete(**kwargs) -> str:
        prompt = kwargs.get('prompt', '')
        return f"Mocked GPT-4o completion for: '{prompt[:70]}...'"

    class MockLightRAG:
        """Simulates LightRAG for testing when the real library is absent."""
        def __init__(self, working_dir: str, llm_model_func: Any, log_level: str):
            self.working_dir = working_dir
            self.llm_func = llm_model_func
            self.log_level = log_level
            print(f"[MockLightRAG Init] Simulating LightRAG.")
            print(f"[MockLightRAG Init]   Working Dir: {working_dir}")
            print(f"[MockLightRAG Init]   LLM Func: {llm_model_func.__name__}")
            if not os.path.exists(working_dir):
                print(f"[MockLightRAG WARN] Simulated working directory does not exist: {working_dir}")

        def query(self, prompt: str, params: Optional[QueryParam] = None) -> str:
            print(f"[MockLightRAG Query] Mode: {params.mode if params else 'default'}")
            print(f"[MockLightRAG Query] Prompt: '{prompt[:100]}...'")
            # Simulate different outputs based on prompt content for testing
            if "error_test" in prompt:
                 print("[MockLightRAG Query] Simulating query error.")
                 raise ValueError("Simulated RAG query error")
            elif "configs" in prompt and "target" in prompt and "Analyze the potential impact" in prompt:
                 print("[MockLightRAG Query] Simulating config analysis response.")
                 return f"Mock Analysis: Config 'CONFIG_HZ_1000' and 'CONFIG_PREEMPT' likely affect target based on simulated knowledge."
            elif prompt:
                 print("[MockLightRAG Query] Simulating knowledge generation response.")
                 # Simulate calling the mock LLM function
                 try:
                     # In a real mock, you might add simulated retrieval context here
                     result = self.llm_func(prompt=prompt) # Call the mock LLM
                 except Exception as e:
                     result = f"Simulated LLM error during mock query: {e}"
                 return result
            else:
                 print("[MockLightRAG Query] Received empty prompt.")
                 return "Mock RAG response: Received empty prompt."

    LightRAGComponent = MockLightRAG # Set the component to use the Mock

# --- Configuration Constants ---
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
# Define a sensible default path relative to this file's location
DEFAULT_LIGHTRAG_WORKING_DIR_REL = "../../../data/lightrag_knowledge_base"
DEFAULT_LIGHTRAG_WORKING_DIR = os.path.abspath(os.path.join(CURRENT_DIR, DEFAULT_LIGHTRAG_WORKING_DIR_REL))

# Get configuration from environment variables or use defaults
LIGHTRAG_WORKING_DIR = os.environ.get("LIGHTRAG_WORKING_DIR", DEFAULT_LIGHTRAG_WORKING_DIR)
DEFAULT_SEARCH_MODE = os.environ.get("LIGHTRAG_SEARCH_MODE", "hybrid")
DEFAULT_LLM_MODEL = os.environ.get("LIGHTRAG_LLM_FUNC", "gpt-4o-mini") # Use model name string
DEFAULT_LOG_LEVEL = os.environ.get("LIGHTRAG_LOG_LEVEL", "WARN")

# --- Tool Definition ---
class OSKnowledgeTool(Tool):
    """
    Tool leveraging a knowledge base (simulated by LightRAG) to provide
    OS-specific knowledge or analyze kernel configuration impacts. Designed
    for integration with Agent-R1 and includes a refined reward function.

    Modes of Operation (specified in 'mode' argument):
    - 'gen_knowledge': Requires 'prompt' argument. Retrieves general knowledge.
    - 'eval_configs': Requires 'configs' (list of strings) and 'target' (string)
                      arguments. Analyzes config impact on the target.
    """
    def __init__(
        self,
        working_dir: str = LIGHTRAG_WORKING_DIR,
        search_mode: str = DEFAULT_SEARCH_MODE,
        llm_model_name: str = DEFAULT_LLM_MODEL, # Use the model name string
        log_level: str = DEFAULT_LOG_LEVEL
        ):
        """
        Initializes the OSKnowledgeTool.

        Args:
            working_dir (str): Path to the LightRAG index and cache directory.
            search_mode (str): Retrieval mode for LightRAG ('bm25', 'dense', 'hybrid').
            llm_model_name (str): Identifier for the LLM to use ('gpt-4o-mini', 'gpt-4o').
            log_level (str): Logging level for the LightRAG instance (e.g., "INFO", "WARN", "ERROR").
        """
        name = "os_knowledge_tool"
        description = (
            "Accesses an OS-specific knowledge base to perform one of two modes: "
            "1) 'gen_knowledge': Retrieves knowledge based on a text 'prompt'. "
            "2) 'eval_configs': Analyzes the impact of a list of 'configs' on a specific 'target'."
            " Specify the desired 'mode' and provide the corresponding required arguments."
        )
        parameters = {
            "type": "object",
            "properties": {
                "mode": {
                    "type": "string",
                    "enum": ["gen_knowledge", "eval_configs"],
                    "description": "Required. The operation mode: 'gen_knowledge' or 'eval_configs'."
                },
                "prompt": {
                    "type": "string",
                    "description": "Required for 'gen_knowledge' mode. The query prompt."
                },
                "configs": {
                    "type": "array",
                    "items": {"type": "string"},
                    "description": "Required for 'eval_configs' mode. List of kernel config names (e.g., 'CONFIG_PREEMPT')."
                },
                "target": {
                    "type": "string",
                    "description": "Required for 'eval_configs' mode. Description of the optimization target (e.g., 'CPU performance')."
                },
                # Optional: Include ground truth during training/evaluation for reward calculation
                "ground_truth": {
                    "type": "string",
                    "description": "(Optional) Expected output string for reward scoring."
                },
            },
            "required": ["mode"] # Primary requirement. Others checked conditionally.
        }
        super().__init__(name, description, parameters)

        # Store configuration needed for lazy initialization
        self.working_dir = working_dir
        self.search_mode = search_mode
        self.llm_model_name = llm_model_name
        self.log_level = log_level
        self._core: Optional[Any] = None # Lazy initialization for the LightRAG core

        # Initial check for working directory existence (non-blocking warning)
        if not os.path.isdir(self.working_dir):
             print(f"[Tool Init WARN] LightRAG working directory does not exist: {self.working_dir}. Initialization will fail later if not created.")

    def _get_core(self) -> Any:
        """Lazy initializes and returns the LightRAG core instance."""
        if self._core is None:
            print(f"[Tool INFO] Lazily initializing LightRAG core...")
            if not os.path.isdir(self.working_dir):
                print(f"[Tool ERROR] LightRAG working directory not found: {self.working_dir}")
                raise FileNotFoundError(f"Required LightRAG working directory not found: {self.working_dir}")

            func_map = {
                "gpt-4o-mini": gpt_4o_mini_complete,
                "gpt-4o": gpt_4o_complete,
            }
            llm_function = func_map.get(self.llm_model_name)
            if not llm_function:
                print(f"[Tool WARN] LLM model name '{self.llm_model_name}' not recognized. Defaulting to gpt-4o-mini.")
                llm_function = gpt_4o_mini_complete

            try:
                self._core = LightRAGComponent( # Use real or mock class
                    working_dir=self.working_dir,
                    llm_model_func=llm_function,
                    log_level=self.log_level
                )
                print(f"[Tool INFO] LightRAG core initialized successfully (Type: {type(self._core).__name__}).")
            except Exception as e:
                print(f"[Tool ERROR] Failed to initialize LightRAG core: {e}")
                import traceback
                traceback.print_exc()
                raise RuntimeError(f"LightRAG core initialization failed: {e}") from e
        return self._core

    def execute(self, args: Dict[str, Any]) -> str:
        """
        Executes the tool based on the specified mode and arguments.

        Args:
            args: Dictionary containing 'mode' and conditional arguments:
                  - 'prompt' (for 'gen_knowledge')
                  - 'configs', 'target' (for 'eval_configs')

        Returns:
            JSON string: {"status": "ok", "output": <result_string>} or
                         {"status": "error", "error": <error_message>}
        """
        mode = args.get("mode")
        output_result = ""
        print(f"[Tool Execute] Received args: {args}")

        try:
            # Get or initialize the LightRAG core instance
            core = self._get_core()

            # Prepare QueryParam - adjust if LightRAG API changes
            # Assuming QueryParam primarily needs the search mode
            query_params = QueryParam(mode=self.search_mode)

            if mode == "gen_knowledge":
                prompt = args.get("prompt", "").strip()
                if not prompt:
                    raise ValueError("Missing required 'prompt' argument for 'gen_knowledge' mode.")
                print(f"[Tool Execute] Running 'gen_knowledge' with prompt: '{prompt[:100]}...'")
                output_result = core.query(prompt, params=query_params)

            elif mode == "eval_configs":
                configs = args.get("configs")
                target = args.get("target", "").strip()
                # Validate inputs specifically for this mode
                if not isinstance(configs, list) or not target:
                    raise ValueError("Missing or invalid required arguments: 'configs' (must be list) and 'target' (must be non-empty string) for 'eval_configs' mode.")
                if not configs:
                     raise ValueError("'configs' list cannot be empty for 'eval_configs' mode.")

                print(f"[Tool Execute] Running 'eval_configs'. Target: '{target}', Configs: {configs[:5]}...")
                # Construct a clear prompt for analysis
                eval_prompt = f"Analyze the potential impact of the following Linux kernel configuration options on the target: '{target}'.\n\nConfigurations to consider:\n"
                eval_prompt += "\n".join([f"- {cfg}" for cfg in configs])
                eval_prompt += "\n\nProvide a concise analysis identifying the most relevant configurations found in the knowledge base and their likely effect (e.g., positive/negative impact, specific tradeoff)."

                # Set target in query_params if LightRAG uses it (example)
                # query_params.target = target
                output_result = core.query(eval_prompt, params=query_params)

            else:
                raise ValueError(f"Invalid 'mode' specified: '{mode}'. Must be 'gen_knowledge' or 'eval_configs'.")

            # Ensure the output is always a string
            output_str = str(output_result) if output_result is not None else ""
            print(f"[Tool Execute] Success. Output length: {len(output_str)}")
            return json.dumps({"status": "ok", "output": output_str}, ensure_ascii=False)

        except Exception as e:
            import traceback
            error_type = type(e).__name__
            error_msg = str(e)
            tb_str = traceback.format_exc()
            print(f"[Tool ERROR] Execution failed - Mode: {mode}, Error: {error_type}: {error_msg}")
            print(f"Traceback:\n{tb_str}")
            return json.dumps({"status": "error", "error": f"{error_type}: {error_msg}"}, ensure_ascii=False)

    def calculate_reward(self, args: Dict[str, Any], result: str) -> float:
        """
        Calculates a reward signal based on the tool's execution result.
        Focuses on successful execution and basic content quality heuristics.
        Optionally incorporates ground truth comparison if 'ground_truth' is in args.

        Args:
            args: The arguments passed to the execute method for this specific call.
            result: The JSON string returned by the execute method.

        Returns:
            A float reward value, typically clipped between -1.0 and 1.0.
        """
        reward = BONUS_BASE_SUCCESS # Start reward at 0.0

        try:
            # Attempt to parse the JSON result string
            result_data = json.loads(result)
            status = result_data.get("status")
            output_content = result_data.get("output", "") if status == "ok" else ""
            error_msg = result_data.get("error", "") if status == "error" else ""

            # 1. Handle Execution Status
            if status == "error":
                print(f"[Reward Calc] Tool execution failed: {error_msg}. Applying penalty.")
                reward += PENALTY_JSON_ERROR # Use a defined penalty constant, e.g., -0.2
                # No further positive rewards if execution failed
                return max(-1.0, min(reward, 1.0)) # Clip and return early

            elif status == "ok":
                 # Base reward for successful execution
                 # reward += 0.1 # Small bonus for just succeeding

                 # 2. Content Quality - Length Heuristic
                 word_count = len(output_content.split())
                 if word_count < 5: # Penalize very short potentially unhelpful answers
                     reward -= 0.05
                     print(f"[Reward Calc] Penalty: Output too short ({word_count} words).")
                 elif word_count > 500: # Penalize very long outputs (might indicate rambling/off-topic)
                     reward -= 0.05
                     print(f"[Reward Calc] Penalty: Output too long ({word_count} words).")
                 else:
                     reward += 0.05 # Small bonus for reasonable length

                 # 3. Content Quality - Ground Truth Overlap (Optional)
                 ground_truth = args.get("ground_truth")
                 if ground_truth and isinstance(ground_truth, str) and output_content:
                     try:
                         output_tokens = set(output_content.lower().split())
                         gt_tokens = set(ground_truth.lower().split())
                         intersection = len(output_tokens.intersection(gt_tokens))
                         union = len(output_tokens.union(gt_tokens))
                         jaccard = intersection / union if union > 0 else 0
                         # Scale the reward contribution (e.g., max +0.4 for perfect overlap)
                         overlap_reward = jaccard * 0.4
                         reward += overlap_reward
                         print(f"[Reward Calc] Bonus: Ground truth Jaccard={jaccard:.2f}, Reward component={overlap_reward:.2f}")
                     except Exception as e:
                         print(f"[Reward Calc WARN] Error calculating ground truth overlap: {e}")
                 elif ground_truth:
                     print("[Reward Calc INFO] Ground truth provided but output content is empty.")


            else:
                 print(f"[Reward Calc WARN] Unknown status in result JSON: {status}")
                 reward += PENALTY_UNEXPECTED_ERROR # Penalize unexpected status

        except json.JSONDecodeError:
            print("[Reward Calc ERROR] Failed to parse result JSON. Applying penalty.")
            reward += PENALTY_JSON_ERROR # Heavy penalty for non-JSON output
        except Exception as e:
            print(f"[Reward Calc ERROR] Unexpected error during reward calculation: {e}")
            import traceback
            traceback.print_exc()
            reward += PENALTY_UNEXPECTED_ERROR # Generic penalty for calculation errors

        # Final clipping
        final_reward = max(-1.0, min(reward, 1.0))
        print(f"[Reward Calc] Args: {args.get('mode')}, Status: {status if 'status' in locals() else 'N/A'}, Final Reward: {final_reward:.3f}")
        return final_reward

    def batch_execute(self, args_list: List[Dict[str, Any]]) -> List[str]:
        """
        Executes a batch of tool calls sequentially.

        Args:
            args_list: A list of argument dictionaries for each call.

        Returns:
            A list of JSON result strings.
        """
        # Placeholder: Implement true batching if LightRAG supports it.
        print(f"[Tool Batch Execute] Processing {len(args_list)} requests sequentially.")
        return [self.execute(args) for args in args_list]

# --- Example Usage ---
if __name__ == '__main__':
    print("--- Testing OSKnowledgeTool ---")

    # Test Initialization (will use mock if lightrag is not installed)
    # Ensure the default working directory exists or change it, e.g., to './temp_rag_data'
    test_wd = './temp_rag_data_os_knowledge'
    os.makedirs(test_wd, exist_ok=True)
    print(f"Using test working directory: {test_wd}")

    tool = OSKnowledgeTool(working_dir=test_wd, llm_model_name="gpt-4o-mini") # Using default search mode

    # Test Case 1: Generate Knowledge - Success
    print("\n--- Test Case 1: Generate Knowledge (Success) ---")
    args1 = {"mode": "gen_knowledge", "prompt": "Explain the PREEMPT_RT patch in Linux."}
    result1 = tool.execute(args1)
    print(f"Args: {args1}")
    print(f"Result: {result1}")
    reward1 = tool.calculate_reward(args1, result1)
    print(f"Reward: {reward1}")

    # Test Case 2: Evaluate Configs - Success
    print("\n--- Test Case 2: Evaluate Configs (Success) ---")
    args2 = {
        "mode": "eval_configs",
        "configs": ["CONFIG_PREEMPT_RT", "CONFIG_HZ_1000", "CONFIG_NO_HZ_FULL", "CONFIG_DEBUG_INFO"],
        "target": "real-time audio processing latency"
    }
    result2 = tool.execute(args2)
    print(f"Args: {args2}")
    print(f"Result: {result2}")
    reward2 = tool.calculate_reward(args2, result2)
    print(f"Reward: {reward2}")

    # Test Case 3: Missing Argument for Mode
    print("\n--- Test Case 3: Missing Argument ---")
    args3 = {"mode": "gen_knowledge"} # Missing 'prompt'
    result3 = tool.execute(args3)
    print(f"Args: {args3}")
    print(f"Result: {result3}")
    reward3 = tool.calculate_reward(args3, result3)
    print(f"Reward: {reward3}")

    # Test Case 4: Invalid Mode
    print("\n--- Test Case 4: Invalid Mode ---")
    args4 = {"mode": "invalid_mode", "prompt": "test"}
    result4 = tool.execute(args4)
    print(f"Args: {args4}")
    print(f"Result: {result4}")
    reward4 = tool.calculate_reward(args4, result4)
    print(f"Reward: {reward4}")

    # Test Case 5: Evaluate Configs with Ground Truth Reward
    print("\n--- Test Case 5: Evaluate Configs with Ground Truth ---")
    args5 = {
        "mode": "eval_configs",
        "configs": ["CONFIG_PREEMPT_RT", "CONFIG_HZ_1000", "CONFIG_DEBUG_INFO"],
        "target": "real-time latency",
        "ground_truth": "CONFIG_PREEMPT_RT and CONFIG_HZ_1000 are highly relevant for real-time latency. CONFIG_DEBUG_INFO is generally negative."
    }
    # Simulate a plausible output (adjust if using real RAG)
    simulated_output5 = "Analysis result for target 'real-time latency': Config 'CONFIG_PREEMPT_RT' seems relevant based on internal knowledge. Config 'CONFIG_HZ_1000' also seems relevant."
    simulated_result5 = json.dumps({"status": "ok", "output": simulated_output5})
    print(f"Args: {args5}")
    print(f"Simulated Result: {simulated_result5}")
    reward5 = tool.calculate_reward(args5, simulated_result5)
    print(f"Reward: {reward5}")

    # Test Case 6: Simulated RAG Error
    print("\n--- Test Case 6: Simulated RAG Error ---")
    args6 = {"mode": "gen_knowledge", "prompt": "trigger_error_test"}
    result6 = tool.execute(args6)
    print(f"Args: {args6}")
    print(f"Result: {result6}")
    reward6 = tool.calculate_reward(args6, result6)
    print(f"Reward: {reward6}")

    # Clean up mock directory if created
    # import shutil
    # if os.path.exists(test_wd) and 'mock' in type(tool._core).__name__.lower():
    #     print(f"Cleaning up test directory: {test_wd}")
    #     shutil.rmtree(test_wd)

# --- END OF FILE os_knowledge_tool.py ---