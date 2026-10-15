# SPDX-License-Identifier: GPL-3.0-or-later
#
# XANESNET
#
# Authors:  Hendrik Junkawitsch, Tom J. Penfold, Thomas J. Pope, C. D. Rankine, B. Li
#
# This program is free software: you can redistribute it and/or modify it under the terms of the
# GNU General Public License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful, but WITHOUT ANY WARRANTY; without
# even the implied warranty of MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# General Public License for more details.
#
# You should have received a copy of the GNU General Public License along with this program.
# If not, see <https://www.gnu.org/licenses/>.
#
# Citations:
#   Junkawitsch et al., "XANESNET: A Modular, Extensible, and Flexible Machine Learning Framework for Spectroscopy."
#   Rankine et al., "A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra."
#   Rankine et al., "Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network."
#   Penfold et al., "Field-Aware Energy-Conditioned Message Passing Neural Networks for Absorber-Centred Modelling of X-ray Spectroscopy."
#   Penfold et al., "A deep neural network for valence-to-core X-ray emission spectroscopy."
#   Falbo et al., "On the Analysis of X-ray Absorption Spectra for Polyoxometallates."
#   Madkhali et al., "Enhancing the Analysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump X-ray Probe Experiments."
#   Madkhali et al., "The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy."

"""Command-line interface dispatcher for XANESNET (train / infer / analyze)."""

import sys

from xanesnet.utils.exceptions import ConfigError

LOGO = r"""
////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//     ██╗  ██╗ █████╗ ███╗   ██╗███████╗███████╗███╗   ██╗███████╗████████╗      //
//     ╚██╗██╔╝██╔══██╗████╗  ██║██╔════╝██╔════╝████╗  ██║██╔════╝╚══██╔══╝      //
//      ╚███╔╝ ███████║██╔██╗ ██║█████╗  ███████╗██╔██╗ ██║█████╗     ██║         //
//      ██╔██╗ ██╔══██║██║╚██╗██║██╔══╝  ╚════██║██║╚██╗██║██╔══╝     ██║         //
//     ██╔╝ ██╗██║  ██║██║ ╚████║███████╗███████║██║ ╚████║███████╗   ██║         //
//     ╚═╝  ╚═╝╚═╝  ╚═╝╚═╝  ╚═══╝╚══════╝╚══════╝╚═╝  ╚═══╝╚══════╝   ╚═╝         //
//                                                                                //
////////////////////////////////////////////////////////////////////////////////////
//                                                                                //
//                         Deep Learning for Spectroscopy                         //
//                                                                                //
////////////////////////////////////////////////////////////////////////////////////
    """

HELP = """Usage: python xanesnet/cli.py <command> [options]

Commands (mutually exclusive):
    train    Train a model using a configuration file.
        Arguments:
            -i, --in_file       Path to input YAML configuration file. (Required)
            -o, --out_dir       Path to output directory. (Optional, default: ./runs )
            -n, --name          Name for the training run used for logging and saving (Optional).
            -t, --tensorboard   Whether to write training metrics to TensorBoard logs (Optional).
            --dry-run           Run one real training epoch and save model_profile.json (Optional).
            -y, --yes           Automatically answer yes to confirmation prompts (Optional).

    infer    Run inference on data using a trained model.
        Arguments:
            -i, --in_file       Path to input YAML configuration file. (Required)
            -m, --in_model      Path to a trained model .pth file. (Required)
            -o, --out_dir       Path to output directory. (Optional, default: ./runs )
            -n, --name          Name for the inference run used for logging and saving (Optional).
            -y, --yes           Automatically answer yes to confirmation prompts (Optional).

    analyze  Analyze predictions from inference runs.
        Arguments:
            -i, --in_file           Path to input YAML configuration file. (Required)
            -r, --inference-runs    Path(s) to inference run directories. Repeatable and/or space separated. (Required)
            -d, --prediction-names  Display names for inference run directories, in the same order as -r. Repeatable and/or space separated. (Optional)
            -o, --out_dir           Path to output directory. (Optional, default: ./runs )
            -n, --name              Name for the analysis run used for logging and saving (Optional).
            -y, --yes               Automatically answer yes to confirmation prompts (Optional).
"""

TRAIN = r"""
 ____ ____ ____ ____ ____ 
||T |||R |||A |||I |||N ||
||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|
"""

INFER = r"""
 ____ ____ ____ ____ ____ 
||I |||N |||F |||E |||R ||
||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|
"""

ANALYZE = r"""
 ____ ____ ____ ____ ____ ____ ____ 
||A |||N |||A |||L |||Y |||Z |||E ||
||__|||__|||__|||__|||__|||__|||__||
|/__\|/__\|/__\|/__\|/__\|/__\|/__\|
"""

################################################################################
############################## PROGRAM STARTS HERE #############################
################################################################################


def main(args: list[str]) -> None:
    """Dispatch to the train, infer, or analyze sub-command.

    Prints the XANESNET logo and routes to the appropriate sub-command entry
    point based on the first positional argument.

    Args:
        args: Raw command-line argument strings (typically ``sys.argv[1:]``).

    Raises:
        ConfigError: If an unrecognized sub-command is supplied.
    """
    print(LOGO)

    if len(args) == 0 or args[0] in ["-h", "--help"]:
        print(HELP)
        sys.exit(0)

    command = args[0]
    remaining = args[1:]

    if command == "train":
        from xanesnet.train import main

        print(TRAIN)

        main(remaining)

    elif command == "infer":
        from xanesnet.infer import main

        print(INFER)

        main(remaining)

    elif command == "analyze":
        from xanesnet.analyze import main

        print(ANALYZE)

        main(remaining)
    else:
        raise ConfigError(f"Incorrect mode: {command}.")


def console_main() -> None:
    """Entry point for the installed ``xanesnet`` console command."""
    main(sys.argv[1:])


if __name__ == "__main__":
    console_main()
