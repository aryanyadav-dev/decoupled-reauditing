"""Global configuration for Decoupled Re-Auditing.

Regime selection by precedence:
1. DRA_SMOKE_TEST=1 forces SMOKE (overrides all)
2. DRA_REGIME env var selects SMOKE / FAST / ABLATION / FULL
3. Default: FULL

Locked experiment plan:
- Exp 1, Exp 2: FULL (300 problems, k=3, 4 generations)
- Exp 3: ABLATION (100 problems, k=3, 2 generations)
- Fast testing: FAST (150 problems, k=3, 3 generations)
- Smoke tests: SMOKE (8 problems, k=2, 1 generation)
"""

import os

SMOKE_TEST = os.getenv("DRA_SMOKE_TEST", "1") == "1"
REGIME_NAME = os.getenv("DRA_REGIME", "FULL").upper()

MODEL_ID = "mistralai/Mistral-7B-Instruct-v0.3"
LLM_JUDGE_ID = "mistralai/Mistral-7B-Instruct-v0.3"
JUDGE_ID = "peiyi9979/math-shepherd-mistral-7b-prm"

TEMPERATURE = 0.8
MAX_NEW_TOKENS_POLICY = 256  # Reduced from 400; GSM8K traces fit in 256 tokens, ~25% speedup
MAX_NEW_TOKENS_JUDGE = 512
LORA_R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.05
LOAD_IN_4BIT = True
SEED = int(os.getenv("DRA_SEED", "42"))  # Configurable seed for multi-seed runs
PRM_THRESHOLD = 0.5
MAX_WALL_CLOCK_SEC = 39600
GEN_BATCH_SIZE = 16  # Number of problems to batch per model.generate() call for speedup (increased from 8)

REGIMES = {
    "SMOKE": dict(NUM_PROBLEMS=8, EVAL_N=8, PROBE_N=8, K_SAMPLES=2, NUM_GENERATIONS=2, TRAIN_STEPS=5),  # 2 gens to test fine-tune -> sample transition
    "FAST": dict(NUM_PROBLEMS=150, EVAL_N=50, PROBE_N=50, K_SAMPLES=3, NUM_GENERATIONS=3, TRAIN_STEPS=100),
    "ABLATION": dict(NUM_PROBLEMS=100, EVAL_N=100, PROBE_N=50, K_SAMPLES=3, NUM_GENERATIONS=2, TRAIN_STEPS=50),
    "FULL": dict(NUM_PROBLEMS=300, EVAL_N=300, PROBE_N=120, K_SAMPLES=3, NUM_GENERATIONS=4, TRAIN_STEPS=200),
}


def active_regime():
    """Select active regime based on environment variables.
    
    Precedence:
    1. DRA_SMOKE_TEST=1 -> SMOKE (overrides all)
    2. DRA_REGIME env var -> specified regime
    3. Default -> FULL
    """
    if SMOKE_TEST:
        name = "SMOKE"
    elif REGIME_NAME in REGIMES:
        name = REGIME_NAME
    else:
        print(f"[config] WARNING: Unknown DRA_REGIME={REGIME_NAME}, falling back to FULL")
        name = "FULL"
    
    vals = dict(REGIMES[name])
    vals["REGIME"] = name
    return vals


_ACTIVE = active_regime()
REGIME = _ACTIVE["REGIME"]
NUM_PROBLEMS = _ACTIVE["NUM_PROBLEMS"]
EVAL_N = _ACTIVE["EVAL_N"]
PROBE_N = _ACTIVE["PROBE_N"]
K_SAMPLES = _ACTIVE["K_SAMPLES"]
NUM_GENERATIONS = _ACTIVE["NUM_GENERATIONS"]
TRAIN_STEPS = _ACTIVE["TRAIN_STEPS"]
