Datasets
==================



======
XANESX
======

This dataset is compatible with the models :ref:`MLP <mlp>`, :ref:`CNN <cnn>`, :ref:`LSTM <lstm>`,
:ref:`AE-MLP <ae-mlp>`, :ref:`AE-CNN <ae-cnn>`, :ref:`AEGAN-MLP <aegan-mlp>`.
It supports both forward ``train_xyz`` and reverse ``train_xanes`` training modes.

The XANESX dataset is a general-purpose dataset for
representing featurised molecular structures and XANES spectra.
Each sample corresponds to a single molecule and is stored as a
serialized data object with the attributes of input features (``x``), target features (``y``),
and the associated energy grid (``e``). Molecular structures are
converted into fixed-length feature representations
using the descriptor(s) defined in the :ref:`Descriptors <descriptors>` section.
XANES spectra are read from text files and can be optionally transformed during preprocessing.


**Input file:**

* ``type: xanesx``
* ``params`` (dict, optional):

  * ``preload`` (int, ``true``): If true, preload the entire dataset into RAM; otherwise load samples on-the-fly.
  * ``fourier`` (float, ``false``): If true,  apply Fourier transformation to the XANES spectra
  * ``gaussian`` (int, ``false``): If true,  apply Gaussian transformation to the XANES spectra
  * ``widths_eV`` (float, ``[0.5, 1.0, 2.0, 4.0]``): Widths (eV) of the Gaussian basis function
  * ``basis_stride`` (int, ``4``): Stride between Gaussian centers on the energy grid

**Example:**
    .. code-block::

        dataset:
          type: xanesx
          root_path: data/fe/processed_train
          xyz_path: data/fe/xyz_train
          xanes_path: data/fe/xanes_train
          params:
            fourier: True



=====
Graph
=====

This dataset is compatible with the models :ref:`GNN <gnn>`
and supports the forward training mode ``train_xyz`` only.

The Graph dataset represents molecular structures as graphs.
Each molecule is encoded as a graph derived from its atomic geometry,
where atom are represented as nodes and interatomic relationships are represented as edges.

Node features include one-hot–encoded atomic numbers and an indicator specifying the absorber atom.
Edge features are generated using a wACSF-style radial Gaussian expansion,
capturing local geometric information over a fixed radial grid defined by ``r_min``, ``r_max``, and the number of points ``n``.

Graph-level attributes are computed using the descriptor(s) defined in the :ref:`Descriptors <descriptors>` section.
The target features XANES spectra (``y``) and the corresponding energy grid (``e``) are defined as additional global attributes.

**Input file:**

* ``type: graph``
* ``params`` (dict, optional):

  * ``preload`` (int, ``true``): If true, preload the entire dataset into RAM; otherwise load samples on-the-fly.
  * ``n`` (int, ``16``): Number of radial basis functions used to encode each edge.
  * ``r_min`` (float, ``0.0``): Minimum interatomic distance for the radial edge encoding.
  * ``r_max`` (float, ``4.0``): Maximum interatomic distance for the radial edge encoding.
  * ``fourier`` (float, ``false``): If true,  apply Fourier transformation to the XANES spectra.
  * ``gaussian`` (int, ``false``): If true,  apply Gaussian transformation to the XANES spectra.
  * ``widths_eV`` (float, ``[0.5, 1.0, 2.0, 4.0]``): Widths (eV) of the Gaussian basis function.
  * ``basis_stride`` (int, ``4``): Stride between Gaussian centers on the energy grid.

**Example:**
    .. code-block::

        dataset:
          type: graph
          root_path:  data/graph-set/processed_train
          xyz_path: data/graph-set/xyz_train
          xanes_path: data/graph-set/xanes_train
          params:
            r_max: 5.0


=====
E3EEmbed
=====

This dataset is compatible with the E3EEmbed models
and supports the forward training mode ``train_xyz`` only.

The e3eembed dataset represents molecular structures using absorber-centred atomic environments derived directly from Cartesian geometry.
Each structure is defined by its atomic numbers (z), Cartesian coordinates (pos), and a mask indicating valid atoms within the system.

Atoms are treated as nodes, with the absorbing atom explicitly identified (by convention index 0).
Node features are not provided as explicit one-hot encodings; instead, atomic numbers are mapped to learnable embeddings within the model.

Geometric relationships between atoms are not precomputed as edge features.
Instead, relative positions with respect to the absorber atom are constructed on-the-fly, enabling the model to compute interatomic distances, unit vectors, and neighbourhood information dynamically.

This formulation allows the use of equivariant message passing, where rotationally consistent features are built directly from interatomic vectors and radial basis expansions during the forward pass, rather than relying on fixed descriptors such as wACSF.

Optional graph-level attributes may be included using the descriptor(s) defined in the :ref:Descriptors <descriptors> section.
The target XANES spectra (y) and the corresponding energy grid (e) are stored as global attributes associated with each structure.

**Input file:**

* ``type: graph``
* ``params`` (dict, optional):

  * ``preload`` (int, ``true``): If true, preload the entire dataset into RAM; otherwise load samples on-the-fly.
  * ``fourier`` (float, ``false``): If true,  apply Fourier transformation to the XANES spectra.
  * ``gaussian`` (int, ``false``): If true,  apply Gaussian transformation to the XANES spectra.
  * ``widths_eV`` (float, ``[0.5, 1.0, 2.0, 4.0]``): Widths (eV) of the Gaussian basis function.
  * ``basis_stride`` (int, ``4``): Stride between Gaussian centers on the energy grid.

**Example:**
    .. code-block::

        dataset:
          type: graph
          root_path:  data/graph-set/processed_train
          xyz_path: data/graph-set/xyz_train
          xanes_path: data/graph-set/xanes_train
          params:


=========
Multihead
=========

This dataset is compatible with the models :ref:`Multihead-MLP <mh-mlp>`,
:ref:`Multihead-CNN <mh-cnn>`.
It supports both forward ``train_xyz`` and reverse ``train_xanes`` training modes.

The Multihead dataset extends the XANESX dataset to support multi-output (multi-head) learning,
where a single molecular structure is associated with multiple XANES spectra.
The dataset accepts multiple ``xyz_path`` and ``xanes_path`` entries.
Each entry corresponds to a distinct prediction head.
During preprocessing, the dataset automatically matches files with the same stem across all provided paths.

Each sample corresponds to a single molecule and is stored as a
serialized data object with the attributes of input features (``x``), target features (``y``),
the associated energy grid (``e``),  head index (``head_idx``), and head name (``head_name``).

Molecular structures are
converted into fixed-length feature representations using
the descriptor(s) defined in the :ref:`Descriptors <descriptors>` section.
XANES spectra are read from text files and can be optionally transformed
(``fourier`` or ``gaussian`` transformations) during preprocessing.
The head index identifies the output head associated with the spectrum,
and head name derived from the directory from which the spectrum was loaded.

* ``type: multihead``
* ``params`` (dict, optional):

  * ``preload`` (int, ``true``): If true, preload the entire dataset into RAM; otherwise load samples on-the-fly.
  * ``n`` (int, ``16``): Number of radial basis functions used to encode each edge.
  * ``r_min`` (float, ``0.0``): Minimum interatomic distance for the radial edge encoding.
  * ``r_max`` (float, ``4.0``): Maximum interatomic distance for the radial edge encoding.
  * ``fourier`` (float, ``false``): If true,  apply Fourier transformation to the XANES spectra.
  * ``gaussian`` (int, ``false``): If true,  apply Gaussian transformation to the XANES spectra.
  * ``widths_eV`` (float, ``[0.5, 1.0, 2.0, 4.0]``): Widths (eV) of the Gaussian basis function.
  * ``basis_stride`` (int, ``4``): Stride between Gaussian centers on the energy grid.

**Example:**
    .. code-block::

        dataset:
          type: multihead
          root_path: data/multihead/processed_train
          xyz_path:
            - data/multihead/train/xyz1
            - data/multihead/train/xyz2
            - data/multihead/train/xyz3
          xanes_path:
            - data/multihead/train/xanes1
            - data/multihead/train/xanes2
            - data/multihead/train/xanes3
          params:
            preload: False

===========
Transformer
===========

This dataset is compatible with the models :ref:`Transformer <transformer>`
and supports the forward training mode ``train_xyz`` only.

The Transformer dataset can be applied for transformer-based architectures in XANESNET.
Each sample corresponds to a single molecule and is encoded with a set of attributes of
per-atom MACE embeddings (``mace``), auxiliary descriptors (``desc``),
atomic positions (``pos``), atomic weights (``weight``), and padding masks (``mask``),
XANES spectra as target features (``y``) and associated energy grid (``e``) .

Both the MACE and auxiliary descriptors
are configurable via the :ref:Descriptors <descriptors> section.
XANES spectra are read from text files and can be optionally transformed during preprocessing.

* ``type: transformer``
* ``params`` (dict, optional):

  * ``preload`` (int, ``true``): If true, preload the entire dataset into RAM; otherwise load samples on-the-fly.
  * ``fourier`` (float, ``false``): If true,  apply Fourier transformation to the XANES spectra.
  * ``gaussian`` (int, ``false``): If true,  apply Gaussian transformation to the XANES spectra.
  * ``widths_eV`` (float, ``[0.5, 1.0, 2.0, 4.0]``): Widths (eV) of the Gaussian basis function.
  * ``basis_stride`` (int, ``4``): Stride between Gaussian centers on the energy grid.

**Example:**
    .. code-block::

        dataset:
          type: transformer
          root_path:  data/fe/processed_train
          xyz_path: data/fe/xyz_train
          xanes_path: data/fe/xanes_train
          params:
            fourier: True


========
EnvEmbed
========

This dataset is compatible with the models :ref:`EnvEmbed <envembed>`.
It supports the forward training mode ``train_xyz`` only.

The EnvEmbed dataset is designed for environment embedding-based architectures in XANESNET.
Each sample corresponds to a single molecule and is represented with a set of attributes including
descriptor features (``desc``) computed from the molecular structure
(configurable via the :ref:`Descriptors <descriptors>`), length of the descriptor features (``lengths``),
distances of all atoms to the absorber atom (``dist``),
coefficient of Gaussian basis transformation of the spectra (``c*``),
and XANES spectra as target features (``y``).

* ``type: envembed``
* ``params`` (dict, optional):

  * ``preload`` (int, ``true``): If true, preload the entire dataset into RAM; otherwise load samples on-the-fly.
  * ``widths_eV`` (float, ``[0.5, 1.0, 2.0, 4.0]``): Widths (eV) of the Gaussian basis function.
  * ``basis_stride`` (int, ``4``): Stride between Gaussian centers on the energy grid.

**Example:**
    .. code-block::

        dataset:
          type: envembed
          root_path:  data/fe/processed_train
          xyz_path: data/fe/xyz_train
          xanes_path: data/fe/xanes_train
          params:
            preload: True
