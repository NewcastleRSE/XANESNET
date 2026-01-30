"""
XANESNET

This program is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free Software 
Foundation, either Version 3 of the License, or (at your option) any later 
version.

This program is distributed in the hope that it will be useful, but WITHOUT ANY
WARRANTY; without even the implied warranty of MERCHANTABILITY or FITNESS FOR A 
PARTICULAR PURPOSE. See the GNU General Public License for more details.

You should have received a copy of the GNU General Public License along with 
this program.  If not, see <https://www.gnu.org/licenses/>.
"""

###############################################################################
############################### LIBRARY IMPORTS ###############################
###############################################################################


import logging
import requests
import torch
import yaml
import tqdm as tqdm
import numpy as np

from pathlib import Path
from ase import Atoms
from typing import TextIO, List, Any, Dict, Tuple

from torch import Tensor
from torch.hub import load_state_dict_from_url

from xanesnet.models.pre_trained import ModelInfo
from xanesnet.utils.gaussian import GaussianBasis
from xanesnet.utils.mode import Mode
from xanesnet.utils.xanes import XANES


###############################################################################
################################## FUNCTIONS ##################################
###############################################################################


def _unique_path(path: Path, base_name: str) -> Path:
    # returns a unique path from `p`/`base_name`_001, `p`/`base_name`_002,
    # `p`/`base_name`_003, etc.

    n = 0
    while True:
        n += 1
        unique_path = path / (base_name + f"_{n:03d}")
        if not unique_path.exists():
            return unique_path


def _list_files(path: Path, with_ext: bool = True) -> list:
    # returns a list of files (as POSIX paths) found in a directory (`d`);
    # 'hidden' files are always omitted and, if with_ext == False, file
    # extensions are also omitted

    return [
        (f if with_ext else f.with_suffix(""))
        for f in path.iterdir()
        if f.is_file() and not f.stem.startswith(".")
    ]


def _str_to_numeric(str_: str):
    # returns the numeric (floating-point or integer) cast of `str_` if
    # cast is allowed, otherwise returns `str_`

    try:
        return float(str_) if "." in str_ else int(str_)
    except ValueError:
        return str_


def linecount(f: Path) -> int:
    # returns the linecount for a file (`f`)

    with open(f, "r") as f_:
        return len([l for l in f_])


def list_filestems(d: Path) -> list:
    # returns a list of file stems (as strings) found in a directory (`d`);
    # 'hidden' files are always omitted
    return [f.stem for f in _list_files(d)]


def mkdir_output(path: Path, name: str):
    save_path = path / name
    save_path.mkdir(parents=True, exist_ok=True)

    return save_path


def save_models(
    path: Path, models: list, metadata: dict, gauss_basis: GaussianBasis = None
):
    """
    Save trained models, descriptors, metadata, and Gaussian reconstructor to disk.
    For bootstrap and ensemble training, the files are saved in a structured directory
    format.
    """

    mode_suffix = metadata["mode"].replace("train_", "")
    path.mkdir(parents=True, exist_ok=True)
    save_path = _unique_path(
        path, metadata["model"]["type"] + "_" + metadata["scheme"] + "_" + mode_suffix
    )
    save_path.mkdir()

    # Save models(s)
    if len(models) == 1:
        torch.save(models[0].state_dict(), save_path / f"model_weights.pth")
        logging.info("Model saved to disk: %s" % save_path.resolve().as_uri())
    elif len(models) > 1:
        for model in models:
            model_dir = _unique_path(save_path, "model")
            model_dir.mkdir()

            torch.save(model.state_dict(), model_dir / f"model_weights.pth")
            logging.info("Model saved to disk: %s" % model_dir.resolve().as_uri())
    else:
        raise ValueError("Nothing to save.")

    # Save Gaussian reconstructor
    if gauss_basis is not None:
        basis_path = save_path / f"gaussian_basis.pt"
        torch.save(gauss_basis.cpu(), basis_path)
        metadata["dataset"]["params"]["basis_path"] = str(basis_path)

    # Save meta data
    metadata["model_dir"] = str(save_path)
    with open(save_path / "metadata.yaml", "w") as f:
        yaml.dump_all([metadata], f)


def save_predict_result(path, mode, result, dataset, pred_eval, scheme, metadata):
    """
    Save prediction and reconstruction results to disk.
    """
    energy = dataset[0].e.numpy() if dataset[0].e is not None else None
    file_names = dataset.file_names

    # Save xanes prediction result to disk
    if mode in [Mode.XYZ_TO_XANES, Mode.BIDIRECTIONAL]:
        save_path = mkdir_output(path, "xanes_pred")
        if energy is None:
            energy = np.arange(result.xanes_pred[0].shape[-1])

        if scheme.mh_flag and not pred_eval:
            # Multi-head scheme: result shapes = (N, H, M)
            nhead = result.xanes_pred[0].shape[1]
            head_names = metadata["dataset"]["params"]["head_names"]

            for h in range(nhead):
                # Create a directory for each head
                head_dir = save_path / head_names[h]
                head_dir.mkdir(parents=True, exist_ok=True)

            for id_, predict_, std_ in tqdm.tqdm(
                zip(file_names, result.xanes_pred[0], result.xanes_pred[1])
            ):
                for h in range(nhead):
                    # Save each head separately in its directory
                    file_head = save_path / head_names[h] / f"{id_}.txt"
                    with open(file_head, "w") as f:
                        save_xanes_mean(f, XANES(energy, predict_[h]), std_[h])

        elif scheme.recon_flag:
            # AE and AEGON scheme: prediction and reconstruction
            save_path = mkdir_output(path, "xyz_recon")
            for id_, recon_, std_ in tqdm.tqdm(
                zip(file_names, result.xyz_recon[0], result.xyz_recon[1])
            ):
                with open(save_path / f"{id_}.txt", "w") as f:
                    save_xyz_mean(f, recon_, std_)
        else:
            # other schemes: result shape = (N, M)
            for id_, predict_, std_ in tqdm.tqdm(
                zip(file_names, result.xanes_pred[0], result.xanes_pred[1])
            ):
                with open(save_path / f"{id_}.txt", "w") as f:
                    save_xanes_mean(f, XANES(energy, predict_), std_)

    # Save XYZ prediction result to disk
    if mode in [Mode.XANES_TO_XYZ, Mode.BIDIRECTIONAL]:
        save_path = mkdir_output(path, "xyz_pred")

        if scheme.mh_flag and not pred_eval:
            # Multi-head scheme: result shapes = (N, H, M)
            nhead = result.xyz_pred[0].shape[1]
            head_names = metadata["dataset"]["head_names"]

            for h in range(nhead):
                # Create a directory for each head
                head_dir = save_path / head_names[h]
                head_dir.mkdir(parents=True, exist_ok=True)

            for id_, predict_, std_ in tqdm.tqdm(
                zip(file_names, result.xyz_pred[0], result.xyz_pred[1])
            ):
                for h in range(nhead):
                    # Save each head separately in its directory
                    file_head = save_path / head_names[h] / f"{id_}.txt"
                    with open(file_head, "w") as f:
                        save_xyz_mean(f, predict_[h], std_[h])

        elif scheme.recon_flag:
            # AE and AEGON scheme: prediction and reconstruction
            if energy is None:
                energy = np.arange(result.xanes_recon[0].shape[1])
            save_path = mkdir_output(path, "xanes_recon")
            for id_, recon_, std_ in tqdm.tqdm(
                zip(file_names, result.xanes_recon[0], result.xanes_recon[1])
            ):
                with open(save_path / f"{id_}.txt", "w") as f:
                    save_xanes_mean(f, XANES(energy, recon_), std_)

        else:
            # other schemes: result shape = (N, M)
            for id_, predict_, std_ in tqdm.tqdm(
                zip(file_names, result.xyz_pred[0], result.xyz_pred[1])
            ):
                with open(save_path / f"{id_}.txt", "w") as f:
                    save_xyz_mean(f, predict_, std_)


def _create_descriptors_from_meta(config: Dict = None):
    """
    Create and return a list of descriptor instances based on the configuration.
    """
    from xanesnet.creator import create_descriptor

    descriptor_list = []

    for descriptor in config:
        des_type = descriptor["type"]
        params = {k: v for k, v in descriptor.items() if k != "type"}
        descriptor = create_descriptor(des_type, **params)
        descriptor_list.append(descriptor)

    return descriptor_list


def load_descriptors_from_local(path: Path) -> List:
    """
    Load one or more descriptors from a local directory.
    """

    logging.info(f">> Loading descriptors from: {path}")
    config_path = path / "metadata.yaml"
    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    descriptor_config = config["descriptors"]

    return _create_descriptors_from_meta(config=descriptor_config)


def _build_and_load_model(model_config: Dict, weight_path: Path) -> Any:
    from xanesnet.creator import create_model

    """Create a model, load its weights, and set to eval mode."""
    # Create a copy to prevent mutation of the original dictionary
    config = model_config.copy()
    model_type = config.pop("type")

    model = create_model(model_type, **config)
    # Load state_dict from the specific weight path
    state_dict = torch.load(weight_path, map_location=torch.device("cpu"))
    model.load_state_dict(state_dict)
    model.eval()

    return model


def load_model_from_local(path: Path):
    """Loads a single model from a directory."""
    config_path = path / "metadata.yaml"
    weight_path = path / "model_weights.pth"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    return _build_and_load_model(config["model"], weight_path)


def load_models_from_local(path: Path) -> List[Any]:
    """Loads an ensemble of models efficiently from a directory."""
    config_path = path / "metadata.yaml"

    with open(config_path, "r") as f:
        config = yaml.safe_load(f)

    model_config = config["model"]

    model_dirs = sorted([d for d in path.iterdir() if d.is_dir()])
    if not model_dirs:
        raise FileNotFoundError(f"No model subdirectories found in {path}")

    model_list = [
        _build_and_load_model(model_config, model_dir / "model_weights.pth")
        for model_dir in model_dirs
    ]

    return model_list


def _load_config_from_url(url: str) -> Dict[str, Any]:
    response = requests.get(url)
    response.raise_for_status()
    config = yaml.safe_load(response.text)
    # model_config = config.get("model")
    # descriptor_config = config.get("descriptors")

    return config


def load_pretrained_descriptors(name: str):
    """
    Create and return a pre-trained model and its descriptors from the weights
    and configuration files.
    Args:
        name: Name of pretrained model.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: Parameters to override default model or descriptor configuration.
    """
    from xanesnet.models import PretrainedModels

    if not hasattr(PretrainedModels, name):
        raise ValueError(f"Model '{name}' is not available in PretrainedModels.")

    meta: ModelInfo = getattr(PretrainedModels, name)
    config = _load_config_from_url(meta.config_url)
    descriptor_config = config.get("descriptors")

    descriptors = _create_descriptors_from_meta(descriptor_config)

    return descriptors


def _overwrite_config(kwargs: Dict[str, Any], config: Dict[str, Any]) -> None:
    """
    Overrides values in config if matching keys are found in kwargs
    """
    for key in config:
        if key in kwargs:
            config[key] = kwargs[key]


def load_pretrained_model(name: str, **kwargs: Any):
    """
    Create and return a pre-trained model and its descriptors from the weights
    and configuration files.
    Args:
        name: Name of pretrained model.
        progress (bool, optional): If True, displays a progress bar of the
            download to stderr. Default is True.
        **kwargs: Parameters to override default model or descriptor configuration.
    """
    from xanesnet.creator import create_model
    from xanesnet.models import PretrainedModels

    if not hasattr(PretrainedModels, name):
        raise ValueError(f"Model '{name}' is not available in PretrainedModels.")

    meta: ModelInfo = getattr(PretrainedModels, name)
    config = _load_config_from_url(meta.config_url)
    model_config = config.get("model")

    # Overwrite values in configurations
    _overwrite_config(kwargs, model_config)

    # Create model instance
    model = create_model(model_config.get("type"), **model_config.get("params"))

    # Obtain weights file
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    state_dict = load_state_dict_from_url(
        meta.weight_url, progress=True, map_location=device
    )

    # Load weights to the model
    model.load_state_dict(state_dict)

    return model


def load_xyz(xyz_f: TextIO) -> Atoms:
    """
    Load an Atoms object from a .xyz file
    """

    xyz_f_l = xyz_f.readlines()

    # pop the number of atoms `n_ats`
    n_ats = int(xyz_f_l.pop(0))

    # pop the .xyz comment block
    comment_block = xyz_f_l.pop(0)

    # pop the .xyz coordinate block
    coord_block = [xyz_f_l.pop(0).split() for _ in range(n_ats)]
    # atomic symbols or atomic numbers
    ats = np.array([l[0] for l in coord_block], dtype="str")
    # atomic coordinates in .xyz format
    xyz = np.array([l[1:] for l in coord_block], dtype="float32")

    try:
        info = dict(
            [
                [key, _str_to_numeric(val)]
                for key, val in [
                    pair.split(" = ") for pair in comment_block.split(" | ")
                ]
            ]
        )
    except ValueError:
        info = dict()

    try:
        # return Atoms object, assuming `ats` contains atomic symbols
        return Atoms(ats, xyz, info=info)
    except KeyError:
        # return Atoms object, assuming `ats` contains atomic numbers
        return Atoms(ats.astype("uint8"), xyz, info=info)


def save_xyz(xyz_f: TextIO, atoms: Atoms):
    """
    Save an Atoms object in .xyz format
    """

    # write the number of atoms in `atoms`
    xyz_f.write(f"{len(atoms)}\n")
    # write additional info ('key = val', '|'-delimited) from the `atoms.info`
    # dictionary to the .xyz comment block
    for i, (key, val) in enumerate(atoms.info.items()):
        if i < len(atoms.info) - 1:
            xyz_f.write(f"{key} = {val} | ")
        else:
            xyz_f.write(f"{key} = {val}")
    xyz_f.write("\n")
    # write atomic symbols and atomic coordinates in .xyz format
    for atom in atoms:
        fmt = "{:<4}{:>16.8f}{:>16.8f}{:>16.8f}\n"
        xyz_f.write(fmt.format(atom.symbol, *atom.position))

    return 0


def load_xanes(file_path: str) -> Tuple[Tensor, Tensor]:
    """
    Load a XANES object from an FDMNES (.txt) output file
    """
    with open(file_path, "r") as f:
        xanes_f_l = f.readlines()

    # pop the FDMNES header block
    for _ in range(2):
        xanes_f_l.pop(0)

    # pop the XANES spectrum block
    xanes_block = [xanes_f_l.pop(0).split() for _ in range(len(xanes_f_l))]
    # absorption energies
    e = torch.tensor([float(l[0]) for l in xanes_block], dtype=torch.float32)
    # absorption intensities
    m = torch.tensor([float(l[1]) for l in xanes_block], dtype=torch.float32)

    return e, m


def save_xanes(xanes_f: TextIO, xanes: XANES):
    """
    Save a XANES object in FDMNES (.txt) output format
    """

    xanes_f.write(f'{"FDMNES":>10}\n{"energy":>10}{"<xanes>":>12}\n')
    for e_, m_ in zip(*xanes.spectrum):
        fmt = f"{e_:>10.2f}{m_:>15.7E}\n"
        xanes_f.write(fmt.format(e_, m_))

    return 0


def save_xanes_mean(xanes_f: TextIO, xanes: XANES, std):
    """
    Save a mean and standard deviation of XANES object in FDMNES (.txt) output format
    """

    xanes_f.write(f'{"FDMNES"}\n{"energy <xanes> <std>"}\n')
    for e_, m_, std_ in zip(*xanes.spectrum, std):
        fmt = f"{e_:<10.2f}{m_:<15.7E}{std_:<15.7E}\n"
        xanes_f.write(fmt.format(e_, m_, std_))

    return 0


def save_xyz_mean(xyz_f: TextIO, mean, std):
    """
    Save a mean and standard deviation of XANES object in FDMNES (.txt) output format
    """

    xyz_f.write(f'{"<xyz> <std>"}\n')
    for m_, std_ in zip(mean, std):
        fmt = f"{m_:<15.7E}{std_:<15.7E}\n"
        xyz_f.write(fmt.format(m_, std_))

    return 0
