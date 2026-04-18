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

import numpy as np
from ase import Atoms
from pyscf import gto, scf
from tblite.interface import Calculator

from .base import Descriptor
from .registry import DescriptorRegistry

###############################################################################
################################## CLASSES ####################################
###############################################################################


@DescriptorRegistry.register("pdos")
class PDOS(Descriptor):
    """
    A class for transforming a molecular system into a projected density of
    states (pDOS) representation.
    """

    def __init__(
        self,
        descriptor_type: str,
        code: str = "xtb",
        method: str = "GFN2-xTB",
        e_min: float = 20.0,
        e_max: float = 20.0,
        sigma: float = 0.7,
        orb_type: str = "p",
        quad_orb_type: str = "d",
        num_points: int = 200,
        basis: str = "3-21g",
        init_guess: str = "minao",
        max_cycles: int = 0,
        use_charge: bool = False,
        use_spin: bool = False,
        use_quad: bool = False,
        use_occupied: bool = False,
        accuracy: float = 1.0,
        guess: int = 0,
        mixer_damping: float = 0.4,
        save_integrals: int = 0,
        temperature: float = 9.5e-4,
        verbosity: int = 0,
    ):
        """
        Args:
            code (str): Backend code ('xtb' or 'pyscf'). Defaults to 'xtb'.
            method (str): xTB method. Defaults to 'GFN2-xTB'.
            e_min (float): Minimum energy grid point for the pDOS (eV). Defaults to 20.0.
            e_max (float): Maximum energy grid point for the pDOS (eV). Defaults to 20.0.
            sigma (float): FWHM of the Gaussian broadening function (eV). Defaults to 0.7.
            orb_type (str): Orbital type for p-DOS projection. Defaults to 'p'.
            quad_orb_type (str): Orbital type for quadrupole transitions. Defaults to 'd'.
            num_points (int): Number of energy grid points. Defaults to 200.
            basis (str): Basis set for pySCF. Defaults to '3-21g'.
            init_guess (str): Initial guess method for pySCF. Defaults to 'minao'.
            max_cycles (int): Number of SCF cycles. Defaults to 0.
            use_charge (bool): Include charge state. Defaults to False.
            use_spin (bool): Include spin state. Defaults to False.
            use_quad (bool): Include d-orbitals for quadrupole transitions. Defaults to False.
            use_occupied (bool): Use occupied orbitals instead of unoccupied. Defaults to False.
            accuracy (float): Numerical thresholds for SCC. Defaults to 1.0.
            guess (int): Initial guess for wavefunction. Defaults to 0.
            mixer_damping (float): SCC mixer damping. Defaults to 0.4.
            save_integrals (int): Keep integral matrices. Defaults to 0.
            temperature (float): Electronic temperature for filling. Defaults to 9.5e-4.
            verbosity (int): Printout verbosity. Defaults to 0.
        """
        super().__init__(descriptor_type)

        self.code = code
        self.method = method
        self.e_min = e_min
        self.e_max = e_max
        self.num_points = num_points
        self.max_cycles = max_cycles
        self.basis = basis
        self.sigma = sigma
        self.init_guess = init_guess
        self.orb_type = orb_type
        self.quad_orb_type = quad_orb_type
        self.use_spin = use_spin
        self.use_charge = use_charge
        self.use_quad = use_quad
        self.use_occupied = use_occupied
        self.verbosity = verbosity

    def transform(self, system: Atoms, site_index: int | None = 0) -> np.ndarray:
        if site_index is None:
            raise NotImplementedError(
                "PDOS does not support computing descriptors for all sites at once. "
                "The electronic structure calculation is per-molecule; pass a specific site_index."
            )
        if self.code == "xtb":
            return self._transform_xtb(system, site_index)
        elif self.code == "pyscf":
            return self._transform_pyscf(system, site_index)
        else:
            raise ValueError(f"Unknown code: {self.code}")

    def _validate_charge_spin(self, total_electrons: int, charge: int, spin: int):
        if (self.use_spin and not self.use_charge) or (not self.use_spin and self.use_charge):
            raise NotImplementedError(
                "For the p-DOS descriptor, it is not a good idea to only consider overall charge or spin state. "
                "Both should be included simultaneously or not at all."
            )
        if self.use_spin and self.use_charge:
            if (((total_electrons - charge) % 2) == 1) and (spin % 2) == 0:
                raise ValueError("The number of electrons is inconsistent with the spin state you have defined.")
            if (((total_electrons - charge) % 2) == 0) and (spin % 2) == 1:
                raise ValueError("The number of electrons is inconsistent with the spin state you have defined.")

    def _transform_xtb(self, system: Atoms, site_index: int) -> np.ndarray:
        numbers = system.get_atomic_numbers()
        nelectron = int(np.sum(numbers))
        positions = system.get_positions() * 1.8897259886  # Å -> bohr

        # charge / spin
        if self.use_spin and self.use_charge:
            charge = int(system.info.get("q", 0))
            spin = int(system.info.get("s", 0))
        else:
            charge = 0
            spin = 0

        self._validate_charge_spin(nelectron, charge, spin)

        # set up xTB calculator
        calc = Calculator(self.method, numbers, positions, charge, spin)
        calc.set("verbosity", self.verbosity)
        calc.set("max-iter", self.max_cycles)

        try:
            res = calc.singlepoint()
            _ = res.get("energy")  # ensure computed
            coeff = np.square(res.get("orbital-coefficients"))  # AO contribution weights
            # Pick p-channel rows depending on site atom Z
            z0 = int(numbers[site_index])
            # For transition metals and heavier series the AO ordering can put core (0:?) then valence;
            # the original code used slices [6:8] for p and [0:4] for d. Preserve that intent.
            if (21 <= z0 <= 29) or (39 <= z0 <= 47) or (57 <= z0 <= 79) or (89 <= z0 <= 112):
                p_rows = slice(6, 8)
                d_rows = slice(0, 4)
            else:
                p_rows = slice(1, 3)  # lighter elements: p ~ rows 1-2 in that basis mapping
                d_rows = slice(0, 0)  # unused unless quad requested for TMs

            # p-DOS fraction for each MO
            p_dos = np.array([np.sum(coeff[p_rows, i]) / np.sum(coeff[:, i]) for i in range(coeff.shape[1])])

            # MO energies and occupations
            orbe = np.asarray(res.get("orbital-energies")) * 27.211324570273  # eV
            orbo = np.asarray(res.get("orbital-occupations"))

            if self.use_occupied:
                weights = p_dos * np.abs(orbo)
            else:
                weights = p_dos * np.abs(orbo - 2.0)  # unoccupied part

            x = np.linspace(self.e_min, self.e_max, num=self.num_points, endpoint=True)
            pdos_gauss = np.asarray(spectrum(orbe, weights, self.sigma, x), dtype=float)

            if self.use_quad:
                if (21 <= z0 <= 29) or (39 <= z0 <= 47) or (57 <= z0 <= 79) or (89 <= z0 <= 112):
                    d_dos = np.array([np.sum(coeff[d_rows, i]) / np.sum(coeff[:, i]) for i in range(coeff.shape[1])])
                else:
                    raise ValueError("d-orbitals are not considered for these atoms.")

                if self.use_occupied:
                    d_weights = d_dos * np.abs(orbo)
                else:
                    d_weights = d_dos * np.abs(orbo - 2.0)

                ddos_gauss = np.asarray(spectrum(orbe, d_weights, self.sigma, x), dtype=float)
                pdos_gauss = np.concatenate([pdos_gauss, ddos_gauss], axis=0)

        except Exception:
            pdos_gauss = np.full(self.num_points * (2 if self.use_quad else 1), np.nan)

        return pdos_gauss

    def _transform_pyscf(self, system: Atoms, site_index: int) -> np.ndarray:
        mol = gto.Mole()
        mol.atom = atoms_to_pyscf(system)
        mol.basis = self.basis

        # charge / spin
        if self.use_spin and self.use_charge:
            charge = int(system.info.get("q", 0))
            spin = int(system.info.get("s", 0))
        else:
            charge = 0
            spin = 0

        # Build molecule (mol.nelectron available after build)
        mol.build(charge=charge, spin=spin)
        self._validate_charge_spin(mol.nelectron, charge, spin)

        # UHF setup
        mf = scf.UHF(mol)
        mf.init_guess = self.init_guess
        mf.max_cycle = self.max_cycles
        mf.verbose = 0
        mf.kernel()  # proceed even if unconverged

        # MO coefficients and labels
        alpha_ao = mf.mo_coeff[0]  # (nao, nmo_a)
        beta_ao = mf.mo_coeff[1]  # (nao, nmo_b)
        ao_labels = mol.ao_labels()  # e.g. "0 C 1s", "0 C 2px", "1 O 2py", ...

        # Energies (Hartree -> eV) and occupations
        alpha_eps = np.asarray(mf.mo_energy[0]) * 27.211324570273
        beta_eps = np.asarray(mf.mo_energy[1]) * 27.211324570273
        alpha_occ = np.asarray(mf.mo_occ[0])
        beta_occ = np.asarray(mf.mo_occ[1])

        # Normalize squared AO coeffs to percentages per MO
        a_sq = np.square(alpha_ao)
        b_sq = np.square(beta_ao)
        a_pct = a_sq / np.sum(a_sq, axis=0, keepdims=True)
        b_pct = b_sq / np.sum(b_sq, axis=0, keepdims=True)

        # Build masks for AOs on the target site and by orbital type
        parsed = [lbl.split() for lbl in ao_labels]  # [atom_idx, element, ao_type, ...]
        ao_on_site = np.array([int(p[0]) == site_index for p in parsed])
        ao_types = [p[2] for p in parsed]

        def build_channel_mask(substr: str) -> np.ndarray:
            # match if substr (e.g., "p" or "d") appears in AO type string like "2px", "3dxy"
            return np.array([substr in t for t in ao_types])

        p_mask = ao_on_site & build_channel_mask(self.orb_type)
        if self.use_quad:
            d_mask = ao_on_site & build_channel_mask(self.quad_orb_type)

        # Sum contributions over matching AOs for each MO
        alpha_pdos = np.sum(a_pct[p_mask, :], axis=0)
        beta_pdos = np.sum(b_pct[p_mask, :], axis=0)

        if self.use_quad:
            alpha_ddos = np.sum(a_pct[d_mask, :], axis=0)
            beta_ddos = np.sum(b_pct[d_mask, :], axis=0)

        # Choose occupied vs unoccupied
        if self.use_occupied:
            a_eps_final = alpha_eps[: mol.nelec[0] - 1]
            a_pdos_final = alpha_pdos[: mol.nelec[0] - 1]
            b_eps_final = beta_eps[: mol.nelec[1] - 1]
            b_pdos_final = beta_pdos[: mol.nelec[1] - 1]
            if self.use_quad:
                a_ddos_final = alpha_ddos[: mol.nelec[0] - 1]
                b_ddos_final = beta_ddos[: mol.nelec[1] - 1]
        else:
            a_eps_final = alpha_eps[mol.nelec[0] :]
            a_pdos_final = alpha_pdos[mol.nelec[0] :]
            b_eps_final = beta_eps[mol.nelec[1] :]
            b_pdos_final = beta_pdos[mol.nelec[1] :]
            if self.use_quad:
                a_ddos_final = alpha_ddos[mol.nelec[0] :]
                b_ddos_final = beta_ddos[mol.nelec[1] :]

        # Broaden
        x = np.linspace(self.e_min, self.e_max, num=self.num_points, endpoint=True)
        alpha_gE = np.asarray(spectrum(a_eps_final, a_pdos_final, self.sigma, x))
        beta_gE = np.asarray(spectrum(b_eps_final, b_pdos_final, self.sigma, x))
        pdos_gauss = 0.5 * (alpha_gE + beta_gE)

        if self.use_quad:
            d_alpha_gE = np.asarray(spectrum(a_eps_final, a_ddos_final, self.sigma, x))
            d_beta_gE = np.asarray(spectrum(b_eps_final, b_ddos_final, self.sigma, x))
            ddos_gauss = 0.5 * (d_alpha_gE + d_beta_gE)
            pdos_gauss = np.concatenate([pdos_gauss, ddos_gauss], axis=0)

        return pdos_gauss


def atoms_to_pyscf(atoms):
    # This converts system in ASE format into atom format for pySCF
    atom_list = []
    for atom in atoms:
        symbol = atom.symbol
        position = atom.position
        atom_list.append((symbol, tuple(position)))
    return atom_list


def spectrum(E, osc, sigma, x):
    # This Gaussian broadens the partial density of states over a defined
    # energy range and grid spacing.
    gE = []
    for Ei in x:
        tot = 0
        for Ej, os in zip(E, osc):
            tot += os * np.exp(-((((Ej - Ei) / sigma) ** 2)))
        gE.append(tot)
    return gE
