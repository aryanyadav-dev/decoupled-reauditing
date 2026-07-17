"""Global configuration for Decoupled Re-Auditing.

Exactly one active regime is resolved by precedence:
SMOKE_TEST, then PILOT_MODE, else FULL.
"""

import os

SMOKE_TEST = os.getenv("DRA_SMOKE_TEST", "1") == "1"
PILOT_MODE = os.getenv("DRA_PILOT_MODE", "0") == "1"

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
SEED = 42
PRM_THRESHOLD = 0.5
MAX_WALL_CLOCK_SEC = 39600
GEN_BATCH_SIZE = 8  # Number of problems to batch per model.generate() call for speedup

REGIMES = {
    "SMOKE": dict(NUM_PROBLEMS=8, EVAL_N=8, PROBE_N=8, K_SAMPLES=2, NUM_GENERATIONS=1, TRAIN_STEPS=5),
    "PILOT": dict(NUM_PROBLEMS=100, EVAL_N=100, PROBE_N=50, K_SAMPLES=5, NUM_GENERATIONS=2, TRAIN_STEPS=50),
    "FULL": dict(NUM_PROBLEMS=500, EVAL_N=500, PROBE_N=200, K_SAMPLES=5, NUM_GENERATIONS=4, TRAIN_STEPS=200),
}


def active_regime():
    if SMOKE_TEST:
        name = "SMOKE"
    elif PILOT_MODE:
        name = "PILOT"
    else:
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
