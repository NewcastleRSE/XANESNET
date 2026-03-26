"""
XANESNET Energy-Embedded E3NN dataset for absorber-centred e3nn models.

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

import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional

import numpy as np
import torch
from tqdm import tqdm
from ase.io import read

from xanesnet.datasets.base_dataset import BaseDataset
from xanesnet.registry import register_dataset
from xanesnet.utils.io import list_filestems
from xanesnet.utils.mode import Mode


@dataclass
class GraphData:
    z: Optional[torch.Tensor] = None
    pos: Optional[torch.Tensor] = None
    mask: Optional[torch.Tensor] = None
    y: Optional[torch.Tensor] = None
    e: Optional[torch.Tensor] = None
    absorber_index: Optional[torch.Tensor] = None
    charge: Optional[torch.Tensor] = None
    spin: Optional[torch.Tensor] = None
    atom_charges: Optional[torch.Tensor] = None
    atom_spins: Optional[torch.Tensor] = None
    average_baseline: Optional[torch.Tensor] = None
    stem: Optional[str] = None

    def to(self, device):
        for attr in [
            "z",
            "pos",
            "mask",
            "y",
            "e",
            "absorber_index",
            "charge",
            "spin",
            "atom_charges",
            "atom_spins",
            "average_baseline",
        ]:
            val = getattr(self, attr)
            if val is not None and torch.is_tensor(val):
                setattr(self, attr, val.to(device))
        return self

@register_dataset("e3eembed")
class E3EEmbdedDataset(BaseDataset):

    def __init__(
        self,
        root: str,
        xyz_path: List[str] | str = None,
        xanes_path: List[str] | str = None,
        mode: Mode = None,
        descriptors: list = None,
        **kwargs,
    ):
        xyz_path = self._unique_path(xyz_path)
        xanes_path = self._unique_path(xanes_path)

        self.absorber_index = kwargs.get("absorber_index", 0)
        self.require_charge_spin = kwargs.get("require_charge_spin", True)
        self.spin_is_multiplicity = kwargs.get("spin_is_multiplicity", False)

        super().__init__(
            Path(root),
            xyz_path,
            xanes_path,
            mode,
            descriptors,
            **kwargs,
        )

        self._register_config(
            dataset_type="e3eembed",
            absorber_index=self.absorber_index,
            require_charge_spin=self.require_charge_spin,
            spin_is_multiplicity=self.spin_is_multiplicity,
        )

    def set_file_names(self):
        xyz_path = self.xyz_path
        xanes_path = self.xanes_path

        if xyz_path and xanes_path:
            xyz_stems = set(list_filestems(xyz_path))
            xanes_stems = set(list_filestems(xanes_path))
            file_names = sorted(list(xyz_stems & xanes_stems))
        elif xyz_path:
            file_names = sorted(list(set(list_filestems(xyz_path))))
        elif xanes_path:
            file_names = sorted(list(set(list_filestems(xanes_path))))
        else:
            raise ValueError("At least one of xyz_path or xanes_path must be provided.")

        if not file_names:
            raise ValueError("No matching files found in the provided paths.")

        self.file_names = file_names

    def _parse_charge_spin_from_xyz_second_line(self, xyz_file: str, stem: str):
        """
        Parse charge and spin from the second line of an XYZ file.

        Expected format, e.g.:
            q = 0 | s = 0

        Supports integer or floating-point values, including negatives.
        """
        with open(xyz_file, "r") as f:
            lines = f.readlines()

        if len(lines) < 2:
            if self.require_charge_spin:
                raise ValueError(
                    f"XYZ file '{xyz_file}' is too short to contain a charge/spin comment line."
                )
            charge = None if self.default_charge is None else 0
            spin = None if self.default_spin is None else 0
            return charge, spin

        comment = lines[1].strip()

        q_match = re.search(r"\bq\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", comment)
        s_match = re.search(r"\bs\s*=\s*([-+]?\d*\.?\d+(?:[eE][-+]?\d+)?)", comment)

        charge = float(q_match.group(1)) if q_match else 0 
        spin = float(s_match.group(1)) if s_match else 0 

        if self.require_charge_spin:
            if charge is None:
                raise ValueError(
                    f"Could not parse charge from second line of '{xyz_file}'. "
                    f"Found: '{comment}'"
                )
            if spin is None:
                raise ValueError(
                    f"Could not parse spin from second line of '{xyz_file}'. "
                    f"Found: '{comment}'"
                )

        return charge, spin

    def _spin_to_unpaired(self, spin_val):
        """
        Convert parsed spin field to PySCF mol.spin, which is the number of unpaired electrons.

        If your XYZ comment stores multiplicity, use unpaired = multiplicity - 1.
        Otherwise interpret the parsed value directly as number of unpaired electrons.
        """
        if spin_val is None:
            return 0

        if self.spin_is_multiplicity:
            n_unpaired = int(round(float(spin_val))) - 1
        else:
            n_unpaired = int(round(float(spin_val)))

        if n_unpaired < 0:
            raise ValueError(f"Invalid derived number of unpaired electrons {n_unpaired} from spin value {spin_val}")
        return n_unpaired

    def _build_pyscf_mol(self, numbers, positions, total_charge, spin_val):
        from pyscf import gto

        atom_spec = [
            (int(Z), (float(r[0]), float(r[1]), float(r[2])))
            for Z, r in zip(numbers, positions)
        ]

        mol = gto.Mole()
        mol.atom = atom_spec
        mol.unit = "Angstrom"
        mol.basis = "def2-svp" 
        mol.charge = 0 if total_charge is None else int(round(float(total_charge)))
        mol.spin = self._spin_to_unpaired(spin_val)
        mol.verbose = 0
        mol.build(parse_arg=False)
        return mol

    def _make_guess_dm(self, mf, mol):
        """
        Build an initial AO density matrix from the requested guess scheme.
        """
        try:
            dm = mf.init_guess_by_minao()
        except Exception:
            dm = mf.init_guess_by_atom()

        return dm

    def _compute_pyscf_atom_props(
        self,
        numbers: np.ndarray,
        positions: np.ndarray,
        total_charge: Optional[float],
        spin_val: Optional[float],
        stem: str,
    ):
        """
        Compute per-atom charges and spin populations from a PySCF guess density.

        Returns:
            atom_charges_torch, atom_spins_torch
        """
        try:
            from pyscf import dft, scf
        except ImportError as exc:
            msg = "PySCF is not installed, but compute_pyscf_atom_props=True was requested."
            if self.require_pyscf_atom_props:
                raise ImportError(msg) from exc
            print(f"[WARN] {msg} Returning atom_charges=None, atom_spins=None for {stem}")
            return None, None

        try:
            mol = self._build_pyscf_mol(
                numbers=numbers,
                positions=positions,
                total_charge=total_charge,
                spin_val=spin_val,
            )

            mf = dft.UKS(mol)
            mf.xc = "B3LYP" 
            mf.verbose = 0

            dm = self._make_guess_dm(mf, mol)
            s = mf.get_ovlp(mol)

            dm = np.asarray(dm)

            if dm.ndim == 2:
                dm = np.asarray((0.5 * dm, 0.5 * dm))

            if dm.ndim != 3 or dm.shape[0] != 2:
                raise ValueError(
                    f"Expected PySCF guess density of shape (2, nao, nao), got {dm.shape}"
                )

            _, atom_charges = scf.uhf.mulliken_pop(mol, dm, s=s, verbose=0)
            _, atom_spins = scf.uhf.mulliken_spin_pop(mol, dm, s=s, verbose=0)

            atom_charges = np.asarray(atom_charges, dtype=np.float32)
            atom_spins = np.asarray(atom_spins, dtype=np.float32)

            if atom_charges.ndim != 1 or len(atom_charges) != len(numbers):
                raise ValueError(
                    f"Unexpected atom_charges shape {atom_charges.shape} for {stem}"
                )

            if atom_spins.ndim != 1 or len(atom_spins) != len(numbers):
                raise ValueError(
                    f"Unexpected atom_spins shape {atom_spins.shape} for {stem}"
                )

            return (
                torch.tensor(atom_charges, dtype=torch.float32),
                torch.tensor(atom_spins, dtype=torch.float32),
            )

        except Exception as exc:
            msg = f"PySCF guess-density population analysis failed for {stem}: {exc}"
            if self.require_pyscf_atom_props:
                raise RuntimeError(msg) from exc
            print(f"[WARN] {msg}")
            return None, None

    def _compute_average_baseline(self) -> Optional[torch.Tensor]:
        """
        Compute the mean XANES spectrum over all files in this dataset.
        Assumes this dataset corresponds to the training set if used for baseline learning.
        """
        if not self.xanes_path:
            return None

        total = None
        n = 0

        for stem in self.file_names:
            xanes_file = os.path.join(self.xanes_path, f"{stem}.txt")
            _, xanes = self.transform_xanes(xanes_file)

            if xanes is None:
                continue

            xanes = xanes.to(torch.float32)

            if total is None:
                total = torch.zeros_like(xanes)

            total += xanes
            n += 1

        if total is None or n == 0:
            return None

        return total / n

    def process(self):

        average_baseline = self._compute_average_baseline()

        if average_baseline is not None:
            torch.save(
                average_baseline,
                os.path.join(self.processed_dir, "average_baseline.pt"),
            )

        for stem in tqdm(self.file_names, total=len(self.file_names)):
            z = pos = mask = y = e = charge = spin = atom_charges = atom_spins = None

            if self.xyz_path:
                xyz_file = os.path.join(self.xyz_path, f"{stem}.xyz")

                # Parse q/s from raw second line
                charge_val, spin_val = self._parse_charge_spin_from_xyz_second_line(
                    xyz_file, stem
                )

                atoms = read(xyz_file)
                z_np = atoms.numbers.astype(np.int64)
                pos_np = atoms.positions.astype(np.float32)

                z = torch.tensor(z_np, dtype=torch.long)
                pos = torch.tensor(pos_np, dtype=torch.float32)
                mask = torch.ones(len(z_np), dtype=torch.bool)

                charge = (
                    None if charge_val is None
                    else torch.tensor(float(charge_val), dtype=torch.float32)
                )
                spin = (
                    None if spin_val is None
                    else torch.tensor(float(spin_val), dtype=torch.float32)
                )

                if self.require_charge_spin:
                    atom_charges, atom_spins = self._compute_pyscf_atom_props(
                        numbers=z_np,
                        positions=pos_np,
                        total_charge=charge_val,
                        spin_val=spin_val,
                        stem=stem,
                    )

            if self.xanes_path:
                xanes_file = os.path.join(self.xanes_path, f"{stem}.txt")
                e, xanes = self.transform_xanes(xanes_file)
            else:
                xanes = None

            if self.mode == Mode.XANES_TO_XYZ:
                raise NotImplementedError(
                    "e3eembed is intended for XYZ -> XANES forward models."
                )
            else:
                y = xanes

            data = GraphData(
                z=z,
                pos=pos,
                mask=mask,
                y=y,
                e=e,
                absorber_index=torch.tensor(self.absorber_index, dtype=torch.long),
                charge=charge,
                spin=spin,
                atom_charges=atom_charges,
                atom_spins=atom_spins,
                average_baseline=average_baseline,
                stem=stem,
            )
            torch.save(data, os.path.join(self.processed_dir, f"{stem}.pt"))

    def collate_fn(self, batch: list[GraphData]) -> GraphData:
        z_list = [sample.z for sample in batch]
        pos_list = [sample.pos for sample in batch]
        mask_list = [sample.mask for sample in batch]
        y_list = [sample.y for sample in batch]
        e_list = [sample.e for sample in batch]
        charge_list = [sample.charge for sample in batch]
        spin_list = [sample.spin for sample in batch]
        atom_charges_list = [sample.atom_charges for sample in batch]
        atom_spins_list = [sample.atom_spins for sample in batch]
        average_baseline_list = [sample.average_baseline for sample in batch]
        stem_list = [getattr(sample, "stem", None) for sample in batch]

        z = self._safe_pad(z_list, dtype=torch.long)
        pos = self._safe_pad(pos_list, dtype=torch.float32)
        mask = self._safe_pad(mask_list, dtype=torch.bool)
        y = self._safe_stack(y_list, dtype=torch.float32)

        e = e_list[0] if all(x is not None for x in e_list) else None

        charge = self._safe_stack(charge_list, dtype=torch.float32)
        spin = self._safe_stack(spin_list, dtype=torch.float32)

        atom_charges = self._safe_pad(atom_charges_list, dtype=torch.float32)
        atom_spins = self._safe_pad(atom_spins_list, dtype=torch.float32)

        average_baseline = (
           average_baseline_list[0]
            if all(x is not None for x in average_baseline_list)
            else None
        )

        if charge is not None:
            charge = charge.view(-1)
        if spin is not None:
            spin = spin.view(-1)

        data = GraphData(
            z=z,
            pos=pos,
            mask=mask,
            y=y,
            e=e,
            absorber_index=torch.tensor(self.absorber_index, dtype=torch.long),
            charge=charge,
            spin=spin,
            atom_charges=atom_charges,
            atom_spins=atom_spins,
            average_baseline=average_baseline,
            stem=stem_list,
        )
        return data

    @property
    def in_features(self):
        return 1

    @property
    def out_features(self):
        y = self[0].y
        return 0 if y is None else len(y)
