Descriptors
===========

Descriptors featurise atomic structures into fixed-size numerical vectors for
descriptor-based datasets (``dataset_type: descriptor``).

wACSF
-----


The weighted Atom-centered Symmetry Functions (wACSF) [1] descriptor
can be used to represent the local environment near an atom by using a
fingerprint composed of the output of multiple two- and
three-body symmetry functions.
wACSF is an extension of the Atom-centered Symmetry Functions (ACSF) [2]
by applying a weighting scheme to the symmetry functions,
which can account for different types of atomic pairs or neighbor interactions more effectively.
Because of that, wACSFs leads to a significantly better generalisation
performance in the machine learning potential than the large set of conventional ACSFs.

| [1] Jörg Behler, Atom-centered symmetry functions for constructing high-dimensional neural network potentials. J. Chem. Phys., 134(7):074106, (2011).
| [2] M. Gastegger, et al., wACSF—Weighted atom-centered symmetry functions as descriptors in machine learning potentials. J. Chem. Phys., 148 (24): 241709, (2018).

Parameters:

* ``descriptor_type`` (str): must be ``wacsf``
* ``r_min`` (float): minimum radial distance in Angstrom (default ``1.0``)
* ``r_max`` (float): maximum radial cutoff in Angstrom (default ``6.0``)
* ``n_g2`` (int): number of G2 (two-body radial) symmetry functions
  (default ``16``)
* ``n_g4`` (int): number of G4 (three-body angular) symmetry functions
  (default ``32``)
* ``l`` (list of float, or ``null``): lambda values for G4 encoding.
  ``null`` uses ``[1.0, -1.0]`` (default ``null``)
* ``z`` (list of float, or ``null``): zeta values for G4 encoding.
  ``null`` uses ``[1.0]`` (default ``null``)
* ``g2_parameterisation`` (str): G2 grid placement — ``shifted`` or
  ``centred`` (default ``shifted``)
* ``g4_parameterisation`` (str): G4 grid placement — ``shifted`` or
  ``centred`` (default ``centred``)
* ``use_charge`` (bool): append the charge state as an extra scalar
  (default ``false``)
* ``use_spin`` (bool): append the spin state as an extra scalar
  (default ``false``)

Example:

.. code-block:: yaml

   descriptors:
     - descriptor_type: wacsf
       r_min: 1.0
       r_max: 6.0
       n_g2: 16
       n_g4: 32

API Reference: :class:`WACSF <xanesnet.descriptors.wacsf.WACSF>`


MACE
----

The MACE descriptor uses a pre-trained MACE (Many-body Atomic Cluster Expansion) model [1] to compute
per-atom equivariant features as a structural fingerprint.
Message-passing layers propagate geometric information through the local
environment; optionally only rotationally invariant components are retained.

[1] Ilyes. B, et al., “MACE: Higher Order Equivariant Message Passing
Neural Networks for Fast and Accurate Force Fields”

Parameters:

* ``descriptor_type`` (str): must be ``mace``
* ``invariants_only`` (bool): if ``true``, return only rotationally invariant
  features (default ``false``)
* ``num_layers`` (int): number of MACE message-passing layers to use;
  ``-1`` uses all layers (default ``-1``)
* ``device`` (str): device to use for the MACE model (``cpu`` or ``cuda``) (default ``cpu``)

Example:

.. code-block:: yaml

   descriptors:
     - descriptor_type: mace
       invariants_only: true
       num_layers: -1

API Reference: :class:`MACE <xanesnet.descriptors.mace.MACE>`


RDC
---

The Radial Distribution Curve (RDC) descriptor transforms an atomic system
into a histogram of pairwise internuclear distances between the target site
and all atoms within a radial cutoff.
Distances are discretised on an auxiliary real-space grid and smoothed with
Gaussians to produce a fixed-length radial fingerprint.

Parameters:

* ``descriptor_type`` (str): must be ``rdc``
* ``r_min`` (float): minimum radial grid distance in Angstrom (default
  ``0.0``)
* ``r_max`` (float): maximum radial cutoff in Angstrom (default ``8.0``)
* ``dr`` (float): grid spacing of the auxiliary real-space grid in Angstrom
  (default ``0.01``)
* ``alpha`` (float): Gaussian exponent controlling smoothing (default
  ``10.0``)
* ``use_charge`` (bool): append the charge state as an extra scalar
  (default ``false``)
* ``use_spin`` (bool): append the spin state as an extra scalar
  (default ``false``)

Example:

.. code-block:: yaml

   descriptors:
     - descriptor_type: rdc
       r_min: 0.0
       r_max: 8.0
       dr: 0.01
       alpha: 10.0

API Reference: :class:`RDC <xanesnet.descriptors.rdc.RDC>`


pDOS
----

The projected density of states (pDOS) descriptor computes a Gaussian-broadened
electronic density of states projected onto atomic orbitals of the target site.
Calculations use either xTB (via tblite) or pySCF; p-channel contributions are
always included and d-channel contributions can be appended optionally.

Parameters:

* ``descriptor_type`` (str): must be ``pdos``
* ``code`` (str): backend — ``xtb`` or ``pyscf`` (default ``xtb``)
* ``method`` (str): xTB Hamiltonian (default ``GFN2-xTB``)
* ``e_min`` (float): lower energy bound for the pDOS grid in eV (default
  ``20.0``)
* ``e_max`` (float): upper energy bound for the pDOS grid in eV (default
  ``20.0``)
* ``sigma`` (float): Gaussian broadening width (FWHM) in eV (default
  ``0.7``)
* ``orb_type`` (str): atomic-orbital type for the primary DOS channel
  (default ``p``)
* ``quad_orb_type`` (str): atomic-orbital type for the quadrupole channel
  (default ``d``)
* ``num_points`` (int): number of grid points (default ``200``)
* ``basis`` (str): basis set for pySCF (default ``3-21g``)
* ``init_guess`` (str): initial-guess method for pySCF SCF (default
  ``minao``)
* ``max_cycles`` (int): maximum SCF iterations (default ``0``)
* ``use_charge`` (bool): read and apply charge from ``system.info`` (default
  ``false``)
* ``use_spin`` (bool): read and apply spin from ``system.info`` (default
  ``false``)
* ``use_quad`` (bool): also compute and append d-channel pDOS (default
  ``false``)
* ``use_occupied`` (bool): project onto occupied MOs instead of unoccupied
  (default ``false``)
* ``verbosity`` (int): printout verbosity for xTB (default ``0``)

Example:

.. code-block:: yaml

   descriptors:
     - descriptor_type: pdos
       code: xtb
       e_min: -20.0
       e_max: 20.0
       sigma: 0.7
       num_points: 200

API Reference: :class:`PDOS <xanesnet.descriptors.pdos.PDOS>`


SOAP
----

Smooth Overlap of Atomic Positions (SOAP) encodes the local geometry around a
site using a SOAP power spectrum computed via the dscribe library.
Radial and angular basis functions summarise the neighbour environment within
a cutoff sphere; optional compression modes reduce the feature dimension.

Parameters:

* ``descriptor_type`` (str): must be ``soap``
* ``r_cut`` (float): local environment cutoff radius in Angstrom (default
  ``6.0``)
* ``n_max`` (int): number of radial basis functions (default ``8``)
* ``l_max`` (int): maximum angular momentum quantum number (default ``6``)
* ``sigma`` (float): Gaussian broadening width in Angstrom (default ``1.0``)
* ``species`` (list of int, or ``null``): atomic numbers to treat as distinct
  species. ``null`` includes all elements H (1) through Lr (103) (default
  ``null``)
* ``average`` (str): averaging mode across atomic centres — ``off``,
  ``inner``, or ``outer`` (default ``outer``)
* ``compression_mode`` (str): SOAP compression — ``off``, ``mu2``,
  ``crossover``, or ``mu1nu1``. ``mu2`` gives a compact moment-tensor
  representation (default ``off``)
* ``compression_species_weighting`` (object, or ``null``): species-weighting
  dictionary for species-weighted compression; ``null`` uses dscribe defaults
  (default ``null``)
* ``use_charge`` (bool): append the charge state as an extra scalar
  (default ``false``)
* ``use_spin`` (bool): append the spin state as an extra scalar
  (default ``false``)

Example:

.. code-block:: yaml

   descriptors:
     - descriptor_type: soap
       r_cut: 6.0
       n_max: 8
       l_max: 6
       compression_mode: mu2

API Reference: :class:`SOAP <xanesnet.descriptors.soap.SOAP>`


Direct
------

The direct descriptor loads pre-computed feature vectors from ``.txt`` files on
disk instead of computing descriptors at runtime.
Each structure must carry ``info["sample_id"]``; the corresponding file is
``{sample_id}.txt`` in ``source_dir``, with whitespace-delimited floats and
one row per site.

Parameters:

* ``descriptor_type`` (str): must be ``direct``
* ``source_dir`` (str): path to the directory holding ``.txt`` descriptor
  files (absolute or relative to the working directory)
* ``preload`` (bool): if ``true``, load all ``.txt`` files into memory at
  initialisation (default ``false``)

Example:

.. code-block:: yaml

   descriptors:
     - descriptor_type: direct
       source_dir: ./data/descriptors/
       preload: true

API Reference: :class:`DIRECT <xanesnet.descriptors.direct.DIRECT>`

