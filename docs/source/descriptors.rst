Descriptors
=======================

=======
wACSF
=======

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

**Input file:**

* ``type: wacsf``
* ``params``:

  * ``r_min`` (float, default = 0.0): The minimum radial cutoff distance (in A) around the absorption site.
  * ``r_max`` (float, default = 8.0):  The maximum radial cutoff distance (in A) around the absorption site.
  * ``n_g2`` (int, default = 0): The number of G2 symmetry functions to use for encoding.
  * ``n_g4`` (int, default = 0): The number of G4 symmetry functions to use for encoding.
  * ``l`` (list, default = [1.0, -1.0]): List of lambda values for G4 symmetry function encoding.
  * ``z`` (list, default = [1.0]):  List of zeta values for G4 symmetry function encoding.
  * ``g2_parameterisation`` (str, default = "shifted"): The strategy to use for G2 symmetry function parameterisation; Options: *"shifted"* or *"centred"*.
  * ``g4_parameterisation`` (str, default = "centred"): The strategy to use for G4 symmetry function parameterisation; Options: *"shifted"* or *"centred"*.
  * ``use_charge`` (bool, default = False): If True, includes an additional element in the vector descriptor for the charge state of the complex.
  * ``use_spin`` (bool, default = False):  If True, includes an additional element in the vector descriptor for the spin state of the complex.

**Example:**
    .. code-block::

        descriptor:
          type: wacsf
          params:
            r_min: 1.0
            r_max: 6.0
            n_g2: 16
            n_g4: 32

..
    .. code-block:: python

        ====
        SOAP
        ====

        Smooth Overlap of Atomic Orbitals (SOAP) [1-3] is a descriptor for
        generating a partial power spectrum.
        This implementation uses real (tesseral) spherical
        harmonics as the angular basis set and provides two orthonormalized
        alternatives for the radial basis functions: spherical primitive gaussian
        type orbitals ("gto") or the polynomial basis set ("polynomial").

        | [1] Albert P, et, al., On representing chemi. Phys. Rev. B 87, 184115, (2013)
        | [2] Albert P, et, al., Comparing molecules and solids across structural and alchemical space. Phys. Chem. Chem. Phys. 18, 13754 (2016)
        | [3] Marc O. J. Jäger, et, al., Machine learning hydrogen adsorption on nanoclusters through structural descriptors. npj Comput. Mater., 4, 37 (2018)

        **Input file:**

        * ``type: soap``
        * ``params``:

          * ``r_cut`` (float, default = 6.0): A cutoff for local region in angstroms. Should be bigger than 1 angstrom for the gto-basis.
          * ``n_max`` (int, default = 16):  The number of radial basis functions.
          * ``l_max`` (int, default = 20): The maximum degree of spherical harmonics.
          * ``sigma`` (float, default = 1.0): The standard deviation of the gaussians used to expand the atomic density.
          * ``rbf`` (str, default = "gto"): The radial basis functions to use. Options: "gto" - Spherical gaussian type orbitals defined as :math:`g_{nl}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\beta_{nn'l} r^l e^{-\alpha_{n'l}r^2}` or * "polynomial" - Polynomial basis defined as :math:`g_{n}(r) = \sum_{n'=1}^{n_\mathrm{max}}\,\beta_{nn'} (r-r_\mathrm{cut})^{n'+2}`
          * ``weighting`` (dict, default = None):  Contains the options which control the weighting of the atomic density. Leave unspecified if you do not wish to apply any weighting. The dictionary may contain the following entries:

                    * ``"function"``: The weighting function to use. The following are currently supported:

                            * ``"poly"``: :math:`w(r) = \{ \begin{array}{ll} c(1 + 2 (\frac{r}{r_0})^{3} -3 (\frac{r}{r_0})^{2}))^{m}, \ \text{for } r \leq r_0\\ 0, \ \text{for } r > r_0 \end{array}\\`
                              This function goes exactly to zero at :math:`r=r_0`. If
                              you do not explicitly provide ``r_cut`` in the
                              constructor, ``r_cut`` is automatically set to ``r0``.
                              You can provide the parameters ``c``, ``m`` and ``r0`` as
                              additional dictionary items. For reference see [4]
                            * ``"pow"``: :math:`w(r) = \frac{c}{d + (\frac{r}{r_0})^{m}}`
                              If you do not explicitly provide ``r_cut`` in the
                              constructor, ``r_cut`` will be set as the value at which
                              this function decays to the value given by the
                              ``threshold`` entry in the weighting dictionary (defaults
                              to 1e-2), You can provide the parameters ``c``, ``d``,
                              ``m``, ``r0`` and ``threshold`` as additional dictionary
                              items. For reference see [5].
                            * ``"exp"``: :math:`w(r) = \frac{c}{d + e^{-r/r_0}}`
                              If you do not explicitly provide ``r_cut`` in the
                              constructor, ``r_cut`` will be set as the value at which
                              this function decays to the value given by the
                              ``threshold`` entry in the weighting dictionary (defaults
                              to 1e-2), You can provide the parameters ``c``, ``d``,
                              ``r0`` and ``threshold`` as additional dictionary items.
                    * ``"w0"``: Optional weight for atoms that are directly on top
                      of a requested center. Setting this value to zero essentially
                      hides the central atoms from the output. If a weighting
                      function is also specified, this constant will override it
                      for the central atoms.

          * ``crossover`` (bool, default = True): The strategy to use for G2 symmetry function parameterisation; Options: *"shifted"* or *"centred"*.
          * ``average`` (str, default = "off"): The strategy to use for G4 symmetry function parameterisation; Options: *"shifted"* or *"centred"*.
          * ``species`` (iterable, default = [1,2,3,4]): If True, includes an additional element in the vector descriptor for the charge state of the complex.
          * ``periodic`` (bool, default = False):  If True, includes an additional element in the vector descriptor for the spin state of the complex.
          * ``sparse`` (bool, default = False): If True, includes an additional element in the vector descriptor for the charge state of the complex.
          * ``dtype`` (str, default = "float64"):  If True, includes an additional element in the vector descriptor for the spin state of the complex.

        | [4] Caro, M. Optimizing many-body atomic descriptors for enhanced computational performance of machine learning based interatomic potentials.  Phys. Rev. B, 100, 024112. (2019).
        | [5] Willatt, M, et, al., Feature optimization for atomistic machine learning  yields a data-driven construction of the periodic table of the elements.  Phys. Chem. Chem. Phys., 20, 29661-29668. (2018).

        **Example:**
            .. code-block::

                descriptor:
                    type: soap
                    params:
                       species: [Fe, H, C, O, N, F, P, S, Cl, Br, I, Si, B, Se, As]
                       n_max: 8
                       l_max: 6
                       r_cut: 6.0

=====
RDC
=====

RDC is a descriptor for transforming a molecular system into a Radial (or 'pair')
Distribution Curve.  The RDC is - simplistically - like a histogram
of pairwise internuclear distances discretised over an auxilliary
real-space grid and smoothed out using Gaussians; pairs are made between
the absorption site and all atoms within a defined radial cutoff.

**Input file:**

* ``type: rdc``
* ``params``:

  * ``r_min`` (float, default = 0.0): The minimum radial cutoff distance (in A) around the absorption site.
  * ``r_max`` (float, default = 6.0):  The maximum radial cutoff distance (in A) around the absorption site.
  * ``dr`` (float, default = 0.01): The step size (in A) for the auxilliary real-space grid that the RDC is discretised over.
  * ``alpha`` (float, default = 10.0): A smoothing parameter used in a Gaussian exponent that defines the effective spatial resolution of the RDC.
  * ``use_charge`` (bool, default = False): If True, includes an additional element in the vector descriptor for the charge state of the complex.
  * ``use_spin`` (bool, default = False):  If True, includes an additional element in the vector descriptor for the spin state of the complex.

**Example:**
	.. code-block::

		descriptor:
		    type: rdc
		    params:
		      r_min: 0.0
		      r_max: 8.0
		      dr: 0.01
		      alpha: 10.0
		      use_charge = False
		      use_spin: False

..
    .. code-block:: python
        =====
        MBTR
        =====

        The descriptor implements Many-body Tensor Representation (MBTR) up to :math:`k=3`.
        You can choose which terms to include by providing a dictionary in the
        k1, k2 or k3 arguments. This dictionary should contain information
        under three keys: "geometry", "grid" and "weighting". See the examples
        below for how to format these dictionaries.
        You can use this descriptor for finite and periodic systems. When dealing
        with periodic systems or when using machine learning models that use the
        Euclidean norm to measure distance between vectors, it is advisable to use
        some form of normalization.
        For the geometry functions the following choices are available:

            * :math:`k=1`:
               * "atomic_number": The atomic numbers.
            * :math:`k=2`:
               * "distance": Pairwise distance in angstroms.
               * "inverse_distance": Pairwise inverse distance in 1/angstrom.
            * :math:`k=3`:
               * "angle": Angle in degrees.
               * "cosine": Cosine of the angle.

        For the weighting the following functions are available:

            * :math:`k=1`:
               * "unity": No weighting.
            * :math:`k=2`:
               * "unity": No weighting.
               * "exp": Weighting of the form :math:`e^{-sx}`
               * "inverse_square": Weighting of the form :math:`1/(x^2)`
            * :math:`k=3`:
               * "unity": No weighting.
               * "exp": Weighting of the form :math:`e^{-sx}`
               * "smooth_cutoff": Weighting of the form :math:`f_{ij}f_{ik}`,
                 where :math:`f = 1+y(x/r_{cut})^{y+1}-(y+1)(x/r_{cut})^{y}`

        The exponential weighting is motivated by the exponential decay of screened
        Coulombic interactions in solids. In the exponential weighting the
        parameters *threshold* determines the value of the weighting function after
        which the rest of the terms will be ignored. Either the parameter *scale*
        or *r_cut* can be used to determine the parameter :math:`s`: *scale*
        directly corresponds to this value whereas *r_cut* can be used to
        indirectly determine it through :math:`s=-\log()`:. The meaning of
        :math:`x` changes for different terms as follows:
        * :math:`k=2`: :math:`x` = Distance between A->B
        * :math:`k=3`: :math:`x` = Distance from A->B->C->A.
        The inverse square and smooth cutoff function weightings use a cutoff
        parameter *r_cut*, which is a radial distance after which the rest of
        the atoms will be ignored. For both, :math:`x` means the distance between
        A->B. For the smooth cutoff function, additional weighting key *sharpness*
        can be added, which changes the value of :math:`y`. If not, it defaults to `2`.
        In the grid setup *min* is the minimum value of the axis, *max* is the
        maximum value of the axis, *sigma* is the standard deviation of the
        gaussian broadening and *n* is the number of points sampled on the
        grid.
        If ``flatten=False``, a list of dense np.ndarrays for each k in ascending order
        is returned. These arrays are of dimension (n_elements x n_elements x
        n_grid_points), where the elements are sorted in ascending order by their
        atomic number.
        If ``flatten=True``, a sparse.COO sparse matrix is returned. This sparse matrix
        is of size (n_features,), where n_features is given by
        get_number_of_features(). This vector is ordered so that the different
        k-terms are ordered in ascending order, and within each k-term the
        distributions at each entry (i, j, h) of the tensor are ordered in an
        ascending order by (i * n_elements) + (j * n_elements) + (h * n_elements).
        This implementation does not support the use of a non-identity correlation
        matrix.

        **Input file:**

        * ``type: mbtr``
        * ``params``:

          * | ``k1`` (dict): Setup for the k=1 term. Default:
            | k1 = {
            |    "geometry": {"function": "atomic_number"},
            |    "grid": {"min": 1, "max": 10, "sigma": 0.1, "n": 50}
            | }
          * | ``k2`` (dict): Dictionary containing the setup for the k=2 term. Contains setup for the used geometry function, discretization and weighting function. Default:
            | k2 = {
            |    "geometry": {"function": "inverse_distance"},
            |    "grid": {"min": 0, "max": 1, "n": 50, "sigma": 0.1},
            |    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            | }
          * | ``k3`` (dict): Dictionary containing the setup for the k=3 term. Contains setup for the used geometry function, discretization and weighting function. Default:
            | k1 = {
            |    "geometry": {"function": "cosine"},
            |    "grid": {"min": -1, "max": 1, "n": 50, "sigma": 0.1},
            |    "weighting": {"function": "exp", "scale": 0.5, "threshold": 1e-3},
            | }
          * ``normalize_gaussians`` (bool, default = True): Determines whether the gaussians are normalized to an area of 1. Defaults to True. If False, the normalization factor is dropped and the gaussians have the form. :math:`e^{-(x-\mu)^2/2\sigma^2}`
          * ``normalization`` (str, default = "none"): Determines the method for normalizing the output. The available options are:

                        * "none": No normalization.
                        * "l2_each": Normalize the Euclidean length of each k-term individually to unity.
                        * "n_atoms": Normalize the output by dividing it with the number of atoms in the system. If the system is periodic, the number of atoms is determined from the given unit cell.
                        * "valle_oganov": Use Valle-Oganov descriptor normalization, with system cell volume and numbers of different atoms in the cell.

          * ``flatten`` (bool, default = True):  Whether the output should be flattened to a 1D array. If False, a dictionary of the different tensors is provided, containing the values under keys: "k1", "k2", and "k3":
          * ``species`` (iterable, default = None): The chemical species as a list of atomic numbers or as a list of chemical symbols. Notice that this is not the atomic numbers that are present for an individual system, but should contain all the elements that are ever going to be encountered when creating the descriptors for a set of systems. Keeping the number of chemical speices as low as possible is preferable.
          * ``periodic`` (bool, default = False): Set to true if you want the descriptor output to respect the periodicity of the atomic systems (see the pbc-parameter in the constructor of ase.Atoms).
          * ``sparse`` (bool, default = False):  Whether the output should be a sparse matrix or a dense numpy array.


        **Example:**
            .. code-block::

                descriptor:
                    type: mbtr
                    params:
                       species: [Fe, H, C, O, N, F, P, S, Cl, Br, I, Si, B, Se, As]
                       k2:
                         geometry:
                            function: distance
                         grid:
                            min: 0
                            max: 6
                            n: 100
                            sigma: 0.1
                         weighting:
                            function: exp
                            scale: 0.5
                            threshold: 0.001
                       k3:
                         geometry:
                            function: angle
                         grid:
                            min: 0
                            max: 180
                            n: 100
                            sigma: 0.1
                         weighting:
                            function: exp
                            scale: 0.5
                            threshold: 0.001
                       periodic: False
                       normalization: l2_each


        =====
        LMBTR
        =====

        The descriptor implements Local Many-body Tensor Representation (MBTR) up to :math:`k=3`.
        Notice that the species of the central atom is not encoded in the output,
        but is instead represented by a chemical species X with atomic number 0.
        This allows LMBTR to be also used on general positions not corresponding to
        real atoms. The surrounding environment is encoded by the two- and
        three-body interactions with neighouring atoms. If there is a need to
        distinguish the central species, one can for example train a different
        model for each central species.
        You can choose which terms to include by providing a dictionary in the k2
        or k3 arguments. The k1 term is not used in the local version. This
        dictionary should contain information under three keys: "geometry", "grid"
        and "weighting". See the examples below for how to format these
        dictionaries.
        You can use this descriptor for finite and periodic systems. When dealing
        with periodic systems or when using machine learning models that use the
        Euclidean norm to measure distance between vectors, it is advisable to use
        some form of normalization.
        For the geometry functions the following choices are available:

            * :math:`k=2`:
               * "distance": Pairwise distance in angstroms.
               * "inverse_distance": Pairwise inverse distance in 1/angstrom.
            * :math:`k=3`:
               * "angle": Angle in degrees.
               * "cosine": Cosine of the angle.

        For the weighting the following functions are available:

            * :math:`k=2`:
               * "unity": No weighting.
               * "exp": Weighting of the form :math:`e^{-sx}`
            * :math:`k=3`:
               * "unity": No weighting.
               * "exp": Weighting of the form :math:`e^{-sx}`

        The exponential weighting is motivated by the exponential decay of screened
        Coulombic interactions in solids. In the exponential weighting the
        parameters *threshold* determines the value of the weighting function after
        which the rest of the terms will be ignored and the parameter **scale**
        corresponds to :math:`s`. The meaning of :math:`x` changes for different
        terms as follows:
        * :math:`k=2`: :math:`x` = Distance between A->B
        * :math:`k=3`: :math:`x` = Distance from A->B->C->A.
        In the grid setup *min* is the minimum value of the axis, *max* is the
        maximum value of the axis, *sigma* is the standard deviation of the
        gaussian broadening and *n* is the number of points sampled on the
        grid.
        If ``flatten=False``, a list of dense np.ndarrays for each k in ascending order
        is returned. These arrays are of dimension (n_elements x n_elements x
        n_grid_points), where the elements are sorted in ascending order by their
        atomic number.
        If ``flatten=True``, a sparse.COO is returned. This sparse matrix
        is of size (n_features,), where n_features is given by
        get_number_of_features(). This vector is ordered so that the different
        k-terms are ordered in ascending order, and within each k-term the
        distributions at each entry (i, j, h) of the tensor are ordered in an
        ascending order by (i * n_elements) + (j * n_elements) + (h * n_elements).
        This implementation does not support the use of a non-identity correlation
        matrix.

        **Input file:**

        * ``type: lmbtr``
        * ``params``:

          * | ``k2`` (dict): Dictionary containing the setup for the k=2 term. Contains setup for the used geometry function, discretization and weighting function. Default:
            | k2 = {
            |    "geometry": {"function": "inverse_distance"},
            |    "grid": {"min": 0.1, "max": 2, "sigma": 0.1, "n": 50},
            |    "weighting": {"function": "exp", "scale": 0.75, "threshold": 1e-2}
            | }
          * | ``k3`` (dict): Dictionary containing the setup for the k=3 term. Contains setup for the used geometry function, discretization and weighting function. Default:
            | k1 = {
            |    "geometry": {"function": "angle"},
            |    "grid": {"min": 0, "max": 180, "sigma": 5, "n": 50},
            |    "weighting" = {"function": "exp", "scale": 0.5, "threshold": 1e-3}
            | }
          * ``normalize_gaussians`` (bool, default = True): Determines whether the gaussians are normalized to an area of 1. Defaults to True. If False, the normalization factor is dropped and the gaussians have the form. :math:`e^{-(x-\mu)^2/2\sigma^2}`
          * ``normalization`` (str, default = "none"): Determines the method for normalizing the output. The available options are:

                        * "none": No normalization.
                        * "l2_each": Normalize the Euclidean length of each k-term individually to unity.

          * ``flatten`` (bool, default = True):  Whether the output should be flattened to a 1D array. If False, a dictionary of the different tensors is provided, containing the values under keys: "k1", "k2", and "k3":
          * ``species`` (iterable, default = None): The chemical species as a list of atomic numbers or as a list of chemical symbols. Notice that this is not the atomic numbers that are present for an individual system, but should contain all the elements that are ever going to be encountered when creating the descriptors for a set of systems. Keeping the number of chemical speices as low as possible is preferable.
          * ``periodic`` (bool, default = False): Set to true if you want the descriptor output to respect the periodicity of the atomic systems (see the pbc-parameter in the constructor of ase.Atoms).
          * ``sparse`` (bool, default = False):  Whether the output should be a sparse matrix or a dense numpy array.


        **Example:**

            .. code-block::

                descriptor:
                    type: lmbtr
                    params:
                       species: [Fe, H, C, O, N, F, P, S, Cl, Br, I, Si, B, Se, As]
                       k2:
                         geometry:
                            function: distance
                         grid:
                            min: 0
                            max: 6
                            n: 100
                            sigma: 0.1
                         weighting:
                            function: exp
                            scale: 0.5
                            threshold: 0.001
                       k3:
                         geometry:
                            function: angle
                         grid:
                            min: 0
                            max: 180
                            n: 100
                            sigma: 0.1
                         weighting:
                            function: exp
                            scale: 0.5
                            threshold: 0.001
                       periodic: False
                       normalization: l2_each

        =====
        MSR
        =====

        The MSR descriptor can be used for transforming a molecular system into a Multiple Scattering
        Representation. MSR encodes the local geometry
        around an absorption site in a manner reminescent of the
        path expansion in multiple scattering theory.

        **Input file:**

        * ``type: msr``
        * ``params``:

          * ``r_min`` (float, default = 0.0): The minimum radial cutoff distance (in A) around the absorption site.
          * ``r_max`` (float, default = 8.0):  The maximum radial cutoff distance (in A) around the absorption site.
          * ``n_s2`` (int, default = 0): Two body terms to use for encoding.
          * ``n_s3`` (int, default = 0): Three body terms to use for encoding.
          * ``n_s4`` (int, default = 0): Four body terms to use for encoding.
          * ``n_s5`` (int, default = 0): Five body terms to use for encoding.
          * ``use_charge`` (bool, default = False): If True, includes an additional element in the vector descriptor for the charge state of the complex.
          * ``use_spin`` (bool, default = False):  If True, includes an additional element in the vector descriptor for the spin state of the complex.

        **Example:**
            .. code-block::

                descriptor:
                  type: armsr
                  params:
                    r_min: 1.0
                    r_max: 6.0
                    n_s2: 16
                    n_s3: 32

        =====
        ARMSR
        =====

        The ARMSR descriptor can be used to transform
        a molecular system into a Multiple Scattering Representation.
        AR-MSRs encode the local geometry
        around an absorption site in a manner reminescent of the path expansion in
        multiple scattering theory. Here, in contrast to the MSR vector, we have an
        angular grid as well as radial grid so information is not overly compressed
        In contrast to MSR, we have truncated the expansion to S3, so that the vector
        does not explode in length, but these higher-order terms can easily be
        included.

        **Input file:**

        * ``type: armsr``
        * ``params``:

          * ``r_min`` (float, default = 0.0): The minimum radial cutoff distance (in A) around the absorption site.
          * ``r_max`` (float, default = 8.0):  The maximum radial cutoff distance (in A) around the absorption site.
          * ``n_s2`` (int, default = 0): Two body terms to use for encoding.
          * ``n_s3`` (int, default = 0): Three body terms to use for encoding.
          * ``use_charge`` (bool, default = False): If True, includes an additional element in the vector descriptor for the charge state of the complex.
          * ``use_spin`` (bool, default = False):  If True, includes an additional element in the vector descriptor for the spin state of the complex.

        **Example:**
            .. code-block::

                descriptor:
                  type: armsr
                  params:
                    r_min: 1.0
                    r_max: 6.0
                    n_s2: 16
                    n_s3: 32


=====
pDOS
=====

The partial Density Of States (pDOS) descriptor encodes
relevant electronic information for ML models seeking to simulate X-ray spectroscopy.
This approach uses a minimal basis set in conjunction with the guess (non-optimised) electronic
configuration to extract and then discretised the density of states (DOS) of the absorbing atom to
form the input vector.

The p-DOS descriptor is aimed at capturing the electronic properties,
which directly link to the spectroscopic observ-
able. To supplement this descriptor with nuclear structure information,
the present descriptor can be concatenated with the
wACSF descriptor.

**Input file:**

* ``type: armsr``
* ``params``:

  * ``r_min`` (float, default = 0.0): The minimum radial cutoff distance (in A) around the absorption site.
  * ``r_max`` (float, default = 6.0):  The maximum radial cutoff distance (in A) around the absorption site.
  * ``e_min`` (float, default = -20.0): The minimum energy grid point for the pDOS (in eV)
  * ``e_max`` (float, default = 20.0): The maximum energy grid point for the pDOS (in eV)
  * ``sigma`` (float, default = 0.7): The FWHM of the Gaussian function used to broaden the pDOS obtained from pySCF.W
  * ``num_points`` (float, default = 200): The number of point over which the broadened pDOS is projected.
  * ``basis`` (str, default = "3-21g"): The basis set used by pySCF during developing the pDOS.
  * ``init_guess`` (str, default = "mminao"): The method of the initial guess used by pySCF during generation of the pDOS.
  * ``max_scf_cycles`` (float, default = 0): The number of SCF cycles used by pySCF during develop the pDOS. Smaller numbers will be closer to the raw guess, while larger number will take longer to load. Note, the warnings are suppressed and so it will not tell you if the SCF is converged. Larger numbers make this more likely, but do not gurantee it.
  * ``use_wacsf`` (bool, default = False): If True, the wACSF descriptor for the structure is also generated and concatenated onto the end after the pDOS descriptor.
  * ``n_g2`` (int, default = 0): The number of G2 symmetry functions to use for encoding.
  * ``n_g4`` (int, default = 0): The number of G4 symmetry functions to use for encoding.
  * ``l`` (list, default = [1.0, -1.0]): List of lambda values for G4 symmetry function encoding.
  * ``z`` (list, default = [1.0]):  List of zeta values for G4 symmetry function encoding.
  * ``g2_parameterisation`` (str, default = "shifted"): The strategy to use for G2 symmetry function parameterisation; Options: *"shifted"* or *"centred"*.
  * ``g4_parameterisation`` (str, default = "centred"): The strategy to use for G4 symmetry function parameterisation; Options: *"shifted"* or *"centred"*.
  * ``use_charge`` (bool, default = False): If True, includes an additional element in the vector descriptor for the charge state of the complex.
  * ``use_spin`` (bool, default = False):  If True, includes an additional element in the vector descriptor for the spin state of the complex.

**Example:**
    .. code-block::

        descriptor:
          type: pdos
          params:
            basis: 3-21G
            init_guess: minao
            orb_type: p
            max_scf_cycles: 0
            num_points: 80
            e_min: -10.0
            e_max: 30.0
            sigma: 0.8
            use_wacsf: True
            r_min: 0.5
            r_max: 6.5
            n_g2: 22
            n_g4: 10


