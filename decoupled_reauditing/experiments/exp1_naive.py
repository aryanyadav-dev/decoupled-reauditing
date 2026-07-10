from decoupled_reauditing.experiments.common import run_real_experiment


def main():
    return run_real_experiment("naive", "exp1_naive", "exp1_naive.csv")


if __name__ == "__main__":
    main()

