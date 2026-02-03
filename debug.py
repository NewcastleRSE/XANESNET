from xanesnet.cli import main


def run_debug_train() -> None:
    debug_args = [
        "train",
        "-i",
        "./.github/workflows/inputs/in_mlp.yaml",
        "--save",
        # "--tensorboard",
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


def run_debug_infer() -> None:
    debug_args = [
        "infer",
        "-i",
        "./.github/workflows/inputs/in_mlp_infer.yaml",
        "-m",
        "./runs/2026-01-22_07-46-16_train_mlp_single/models/final.pth",
        # "--tensorboard",
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


if __name__ == "__main__":
    run_debug_train()
