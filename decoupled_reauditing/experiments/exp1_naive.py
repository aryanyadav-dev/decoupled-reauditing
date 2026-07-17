from decoupled_reauditing import config
from decoupled_reauditing.experiments.common import run_real_experiment


def main():
    # Print active regime for logging/verification
    print("="*80)
    print(f"[exp1_naive] Active regime: {config.REGIME}")
    print(f"  NUM_PROBLEMS={config.NUM_PROBLEMS}, K_SAMPLES={config.K_SAMPLES}, NUM_GENERATIONS={config.NUM_GENERATIONS}")
    print(f"  EVAL_N={config.EVAL_N}, PROBE_N={config.PROBE_N}, TRAIN_STEPS={config.TRAIN_STEPS}")
    print(f"  SEED={config.SEED}")
    print("="*80)
    
    return run_real_experiment("naive", "exp1_naive", "exp1_naive.csv")


if __name__ == "__main__":
    main()

