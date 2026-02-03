from xanesnet.cli import main

# DEBUG ENTRY POINTS FOR DEVELOPMENT PURPOSES ONLY


def run_debug_train() -> None:
    debug_args = [
        "train",
        "-i",
        "./configs/in_mlp.yaml",
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
        "./configs/in_mlp_infer.yaml",
        "-m",
        "<insert trained model here>",  # Insert path to trained model (final.pth)
        # "--tensorboard",
    ]

    print("Running in debug mode with the following arguments:")
    print(debug_args)

    main(debug_args)


if __name__ == "__main__":
    run_debug_train()
