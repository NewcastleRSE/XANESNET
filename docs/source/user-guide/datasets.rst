Datasets
========

Datasets convert raw structures and spectra from a datasource into processed
samples for training and inference. They are configured under ``dataset``.
Processed samples are written as ``.pth`` files under ``root``.

Periodic (materials) variants append ``_mp`` to ``dataset_type`` and accept
``num_workers``. Inverse variants append ``_inverse`` and swap descriptors
and spectra so the spectrum is the model input.


descriptor
----------

The descriptor dataset featurises each absorbing site into a fixed-length
vector using the descriptors listed under ``descriptors``
(see :doc:`descriptors`).
It produces :class:`Descriptor Data <xanesnet.datasets.torch.descriptor.DescriptorData>`
for :class:`MLP <xanesnet.models.mlp.mlp.MLP>` model.
Each sample corresponds to one absorbing site and is stored as a
serialized object with these attributes:

* ``x``: model input features
* ``y``: model target features
* ``energies``: energy grid of the spectrum, one value per intensity
* ``sample_id``: sample identifier metadata for one sample or a batch
* ``element``: atomic number of the absorbing site, used by
  element-aware encodings
* ``target_site_index``: index of the absorbing atom in the original
  structure

In the forward type (``descriptor``) the descriptor is the input and the
spectrum is the target. In the inverse type (``descriptor_inverse``) those
roles are swapped.

Parameters:

* ``dataset_type`` (str): must be ``descriptor``, ``descriptor_inverse``,
  ``descriptor_mp``, or ``descriptor_inverse_mp``
* ``root`` (str): directory that stores processed ``.pth`` files
* ``preload`` (bool): load processed samples into RAM when ``true``
  (default ``true``)
* ``skip_prepare`` (bool): reuse existing processed files when ``true``
  (default ``false``)
* ``split_ratios`` (list of float, or ``null``): train/validation split
  fractions; values must sum to ``1.0`` when ``split_indexfile`` is
  ``null`` (default ``[1.0]``)
* ``split_indexfile`` (str or ``null``): optional path to split indices
  (default ``null``)
* ``descriptors`` (list): descriptor blocks applied to each structure
  (see :doc:`descriptors`)
* ``num_workers`` (int or ``null``): worker count for ``_mp`` types;
  ``null`` uses the CPU count (default ``null``)

Example:

.. code-block:: yaml

   dataset:
     dataset_type: descriptor
     root: ./data/processed/toy_data_mlp/
     preload: true
     skip_prepare: false
     split_ratios: [0.8, 0.2]
     descriptors:
       - descriptor_type: wacsf
         r_min: 1.0
         r_max: 6.0
         n_g2: 16
         n_g4: 32

API Reference: :class:`DescriptorDataset <xanesnet.datasets.torch.descriptor.DescriptorDataset>`


descriptor_multihead
--------------------

The descriptor multi-head dataset is the multi-output counterpart of
``descriptor``. Each datasource subdirectory maps to one prediction head.
It produces :class:`Descriptor Multihead Data <xanesnet.datasets.torch.descriptor_multihead.DescriptorMultiheadData>`
for :class:`MultiHead_MLP <xanesnet.models.multihead.mh_mlp.MultiHead_MLP>`
and :class:`MultiHead_CNN <xanesnet.models.multihead.mh_cnn.MultiHead_CNN>` models.
Each sample corresponds to one absorbing site and is stored as a
serialized object with these attributes:

* ``x``: model input features
* ``y``: model target features
* ``energies``: energy grid of the spectrum, one value per intensity
* ``sample_id``: sample identifier metadata for one sample or a batch
* ``element``: atomic number of the absorbing site, used by
  element-aware encodings
* ``head_idx``: which prediction head this sample belongs to, taken
  from the datasource subdirectory


Parameters:

* ``dataset_type`` (str): must be ``descriptor_multihead``,
  ``descriptor_multihead_inverse``, ``descriptor_multihead_mp``, or
  ``descriptor_multihead_inverse_mp``
* ``root`` (str): directory that stores processed ``.pth`` files
* ``preload`` (bool): load processed samples into RAM when ``true``
  (default ``true``)
* ``skip_prepare`` (bool): reuse existing processed files when ``true``
  (default ``false``)
* ``split_ratios`` (list of float, or ``null``): train/validation split
  fractions; values must sum to ``1.0`` when ``split_indexfile`` is
  ``null`` (default ``[1.0]``)
* ``split_indexfile`` (str or ``null``): optional path to split indices
  (default ``null``)
* ``descriptors`` (list): descriptor blocks applied to each structure
  (see :doc:`descriptors`)
* ``num_workers`` (int or ``null``): worker count for ``_mp`` types;
  ``null`` uses the CPU count (default ``null``)

Example:

.. code-block:: yaml

   dataset:
     dataset_type: descriptor_multihead
     root: ./data/processed/toy_data_multihead_mlp/
     preload: true
     skip_prepare: false
     split_ratios: [0.8, 0.2]
     descriptors:
       - descriptor_type: wacsf
         r_min: 1.0
         r_max: 6.0
         n_g2: 16
         n_g4: 32

API Reference: :class:`DescriptorMultiheadDataset <xanesnet.datasets.torch.descriptor_multihead.DescriptorMultiheadDataset>`


envembed
--------

The EnvEmbed dataset builds a target-site-centred environment for each
absorbing atom: per-site descriptors, distances from the absorber, and a
Gaussian spectral basis used to reconstruct the spectrum.
It produces :class:`EnvEmbed Data <xanesnet.datasets.torch.envembed.EnvEmbedData>`
for :class:`EnvEmbed <xanesnet.models.envembed.envembed.EnvEmbed>` model.
Each sample corresponds to one absorbing site and is stored as a
serialized object with these attributes:

* ``descriptor_features``: per-site descriptor vectors, with the
  absorbing site first
* ``distance_features``: distance of each site from the absorber, in
  Angstrom
* ``intensities``: spectrum intensities
* ``energies``: energy grid of the spectrum, one value per intensity
* ``c_star``: Gaussian basis coefficients of the spectrum
* ``lengths``: number of real sites before padding
* ``sample_id``: sample identifier metadata for one sample or a batch
* ``element``: atomic number of the absorbing site, used by
  element-aware encodings
* ``target_site_index``: index of the absorbing atom in the original
  structure
* ``basis``: spectral basis used to rebuild the spectrum

Parameters:

* ``dataset_type`` (str): must be ``envembed`` or ``envembed_mp``
* ``root`` (str): directory that stores processed ``.pth`` files
* ``preload`` (bool): load processed samples into RAM when ``true``
  (default ``true``)
* ``skip_prepare`` (bool): reuse existing processed files when ``true``
  (default ``false``)
* ``split_ratios`` (list of float, or ``null``): train/validation split
  fractions; values must sum to ``1.0`` when ``split_indexfile`` is
  ``null`` (default ``[1.0]``)
* ``split_indexfile`` (str or ``null``): optional path to split indices
  (default ``null``)
* ``widths_eV`` (list of float): Gaussian basis widths in eV (default
  ``[0.2, 1.0, 2.0, 4.0]``)
* ``basis_stride`` (int): energy-grid stride used when creating the
  Gaussian basis (default ``4``)
* ``basis_path`` (str or ``null``): optional path to a serialized spectral
  basis (default ``null``)
* ``env_radius`` (float or ``null``): periodic-neighbour cutoff in
  Angstrom for structures (default ``null``)
* ``descriptors`` (list): descriptor blocks applied to each site
  (see :doc:`descriptors`)
* ``num_workers`` (int or ``null``): worker count for ``envembed_mp``;
  ``null`` uses the CPU count (default ``null``)

Example:

.. code-block:: yaml

   dataset:
     dataset_type: envembed
     root: ./data/processed/toy_data_envembed/
     preload: true
     skip_prepare: false
     split_ratios: [0.8, 0.2]
     widths_eV: [0.2, 1.0, 2.0, 4.0]
     basis_stride: 4
     env_radius: 6.0
     descriptors:
       - descriptor_type: wacsf
         r_min: 1.0
         r_max: 6.0
         n_g2: 16
         n_g4: 32

API Reference: :class:`EnvEmbedDataset <xanesnet.datasets.torch.envembed.EnvEmbedDataset>`


e3ee
----

The e3eembed dataset represents molecular structures using absorber-centred atomic environments derived directly from Cartesian geometry.
Each structure is defined by its atomic numbers (z), Cartesian coordinates (pos), and a mask indicating valid atoms within the system.

Atoms are treated as nodes, with the absorbing atom explicitly identified (by convention index 0).
Node features are not provided as explicit one-hot encodings; instead, atomic numbers are mapped to learnable embeddings within the model.

Geometric relationships between atoms are not precomputed as edge features.
Instead, relative positions with respect to the absorber atom are constructed on-the-fly, enabling the model to compute interatomic distances, unit vectors, and neighbourhood information dynamically.

This formulation allows the use of equivariant message passing, where rotationally consistent features are built directly from interatomic vectors and radial basis expansions during the forward pass, rather than relying on fixed descriptors such as wACSF.

It is the dataset used by :class:`E3EE <xanesnet.models.e3ee.e3ee.E3EE>` model.
Each sample corresponds to one absorbing site and is stored as a
serialized object with these attributes:

* ``x``: atomic numbers of all atoms in the structure
* ``target_site_index``: index of the absorbing atom
* ``edge_src`` / ``edge_dst``: source and destination atom indices of
  the main neighbour graph
* ``edge_weight``: edge lengths in Angstrom
* ``edge_vec``: edge displacement vectors in Angstrom
* ``att_dst``: destination atom indices of the attention graph
* ``att_dist``: absorber-to-atom distances in Angstrom
* ``att_vec``: absorber-to-atom displacement vectors in Angstrom
* ``energies``: energy grid of the spectrum, one value per intensity
* ``intensities``: spectrum intensities
* ``sample_id``: sample identifier metadata for one sample or a batch

When ``use_path_branch`` is ``true``, three-body path fields are also
stored:

* ``path_j`` / ``path_k``: neighbour atom indices of each path
* ``path_r0j`` / ``path_r0k`` / ``path_rjk``: absorber–j, absorber–k,
  and j–k distances in Angstrom
* ``path_cosangle``: cosine of the j–absorber–k angle

Parameters:

* ``dataset_type`` (str): must be ``e3ee`` or ``e3ee_mp``
* ``root`` (str): directory that stores processed ``.pth`` files
* ``preload`` (bool): load processed samples into RAM when ``true``
  (default ``true``)
* ``skip_prepare`` (bool): reuse existing processed files when ``true``
  (default ``false``)
* ``split_ratios`` (list of float, or ``null``): train/validation split
  fractions; values must sum to ``1.0`` when ``split_indexfile`` is
  ``null`` (default ``[1.0]``)
* ``split_indexfile`` (str or ``null``): optional path to split indices
  (default ``null``)
* ``graph_builder`` (object): main message-passing graph (same keys as
  geometrygraph)
* ``att_graph_builder`` (object): attention-graph builder (same keys as
  geometrygraph)
* ``use_path_branch`` (bool): precompute target-site-centred paths
  (default ``false``)
* ``max_paths_per_structure`` (int): maximum target-site paths saved per
  structure (default ``128``)
* ``num_workers`` (int or ``null``): worker count for ``e3ee_mp``;
  ``null`` uses the CPU count (default ``null``)

Example:

.. code-block:: yaml

   dataset:
     dataset_type: e3ee
     root: ./data/processed/toy_data_e3ee/
     preload: true
     skip_prepare: false
     split_ratios: [0.8, 0.2]
     graph_builder:
       graph_builder_type: cov_radius
       cutoff: 5.0
       cov_radii_scale: 2.5
       max_num_neighbors: 50
     att_graph_builder:
       graph_builder_type: radius
       cutoff: 10.0
       max_num_neighbors: 128
     use_path_branch: false
     max_paths_per_structure: 128

API Reference: :class:`E3EEDataset <xanesnet.datasets.torchgeometric.e3ee.E3EEDataset>`

