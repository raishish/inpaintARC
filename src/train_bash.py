from llmtuner import run_exp


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


def main():
    run_exp()


if __name__ == "__main__":
    main()
