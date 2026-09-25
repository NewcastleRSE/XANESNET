Models
======

MLP
---

A Multilayer Perceptron (MLP) is a straightforward architecture composed of stacked, fully connected layers.
It takes a :class:`Descriptor Data <xanesnet.datasets.torch.descriptor.DescriptorData>` as input
and predicts its corresponding spectrum.

The model consists of a customizable sequence of hidden layers followed by a single output layer.
Each hidden layer is composed of a linear (dense) layer, a dropout layer for regularization, 
and a non-linear activation function. 
The size of first hidden layer is determined by the ``hidden_size`` parameter,
and the size of subsequent hidden layers is determined by the ``shrink_rate`` parameter,
which multiplicative reduces the size of the previous layer.
For example, if ``hidden_size: 256`` and ``shrink_rate: 0.5``,
the hidden layers will have sizes ``256``, ``128``, and ``64``.

The output layer is a linear layer that maps the hidden layers to the output spectrum.
The size of the output layer is determined by the ``out_size`` parameter,
which must be equal to the length of the spectrum.

Parameters:

* ``model_type`` (str): must be ``mlp``
* ``in_size`` (int or ``auto``): length of the input features. ``auto``
  computes it from the descriptor at training time
* ``out_size`` (int or ``auto``): length of the output features. ``auto``
  computes it from the target at training time
* ``hidden_size`` (int): size of the first hidden layer (default
  ``226``)
* ``dropout`` (float): dropout rate applied after each hidden layer, from ``0`` to ``1`` (default ``0.1``)
* ``num_hidden_layers`` (int): number of hidden layers excluding the output layer (default ``3``).
* ``shrink_rate`` (float): multiplicative factor applied to the layer width at each depth step (default ``0.5``).
* ``activation`` (str):  name of the activation function for hidden layers
  (default ``prelu``). available values: ``relu``, ``prelu``, ``tanh``,
  ``sigmoid``, ``elu``, ``leakyrelu``, ``selu``, ``silu``, ``gelu``

Example:

.. code-block:: yaml

   model:
     model_type: mlp
     in_size: auto
     out_size: auto
     hidden_size: 512
     dropout: 0.2
     num_hidden_layers: 5
     shrink_rate: 0.2
     activation: prelu

API Reference: :class:`MLP <xanesnet.models.mlp.mlp.MLP>`


E3EE
-----

The Equivariant Graph Neural Network with Energy-Conditioned Attention (E3EENet) is 
an absorber-centric architecture designed to model local 
electronic structure in X-ray absorption processes. 
The model takes a :class:`E3EE Data <xanesnet.datasets.torch.e3ee.E3EEData>` as input
and predicts its corresponding spectrum.

Built upon a tensor-product message-passing framework within the broader class of 
E(3)-equivariant neural networks, 
the model preserves rotational and translational symmetries 
while explicitly encoding the geometric environment of the absorbing atom. 
This equivariant formulation enables a physically consistent representation 
of both radial and angular correlations, 
allowing the network to capture anisotropic scattering and 
local coordination effects with high fidelity. 
By conditioning interactions on the incident energy, 
e3eenet further introduces an adaptive mechanism that modulates 
the contribution of neighbouring atoms as a function of energy, 
providing a natural route to modelling energy-dependent spectral features.

First, an equivariant atomic encoder generates per-atom latent features. 
Second, these equivariant features are converted into rotationally invariant atomwise 
summaries that retain information from scalar and higher-order channels. 
Third, an energy-conditioned attention mechanism constructs an energy-dependent absorber 
representation by attending over all atoms in the local environment. 
Optionally, absorber-centred path terms may be added to capture higher-order geometric effects.

Parameters:

* ``model_type`` (str): must be ``e3ee``
* ``out_size`` (int or ``auto``): number of energy-grid points (output length).
  ``auto`` computes it from the target at training time
* ``max_z`` (int): maximum atomic number, inclusive (default ``100``)
* ``atom_emb_dim`` (int): element embedding dimension (default ``128``)
* ``atom_hidden_dim`` (int): hidden dimension in the equivariant encoder
  (default ``128``)
* ``atom_layers`` (int): number of equivariant interaction blocks
  (default ``3``)
* ``local_cutoff`` (float): message-passing cutoff radius in Angstrom
  (default ``6.0``)
* ``rbf_dim`` (int): number of Gaussian RBF bases for local edge distances
  (default ``32``)
* ``energy_rbf_dim`` (int): number of Gaussian RBF bases for energy embedding
  (default ``48``)
* ``scatter_dim`` (int): intermediate scatter feature dimension for the path
  branch (default ``128``)
* ``latent_dim`` (int): output dimension produced by each active branch
  (default ``128``)
* ``head_hidden_dim`` (int): hidden dimension of the final head MLP and gated
  fusion MLPs (default ``128``)
* ``e3nn_irreps`` (str): node irreps for the equivariant encoder
  (default ``64x0e + 32x1o + 16x2e``)
* ``e3nn_irreps_message`` (str): message irreps inside each interaction block
  (default ``16x0e + 8x1o + 4x2e``)
* ``e3nn_lmax`` (int): maximum ``l`` for spherical harmonics in the encoder
  (default ``2``)
* ``out_mlp_layers`` (int): number of layers in the final head MLP
  (default ``3``)
* ``use_invariant_branch`` (bool): enable the invariant target-site branch
  (default ``true``)
* ``use_attention_branch`` (bool): enable the invariant atom-attention branch
  (default ``true``)
* ``use_equivariant_branch`` (bool): enable the late equivariant target-site
  head (default ``true``)
* ``use_eq_attention_branch`` (bool): enable the equivariant atom-attention
  branch (default ``false``)
* ``use_conv_branch`` (bool): enable the invariant SchNet/PaiNN convolution
  branch (default ``false``)
* ``use_eq_conv_branch`` (bool): enable the equivariant NequIP/MACE
  convolution branch (default ``false``)
* ``use_path_branch`` (bool): enable the 3-body path-scattering branch
  (default ``false``)
* ``fusion_mode`` (str): how active branches are combined before the final
  head. ``cat`` concatenates them; ``gated`` uses energy-conditioned soft
  gates (default ``cat``)
* ``residual_scale_init`` (float): initial value of learnable residual scales
  (default ``0.1``)
* ``attention_heads`` (int): number of attention heads in attention branches
  (default ``4``)
* ``attention_rbf_dim`` (int): number of RBF bases for attention-graph
  distances (default ``16``)
* ``attention_lmax`` (int): maximum ``l`` for spherical harmonics in
  attention/conv branches (default ``2``)
* ``attention_irreps`` (str): output irreps of equivariant attention/conv
  branches (default ``32x0e + 16x1o + 8x2e``)
* ``att_cutoff`` (float): attention-graph cutoff radius in Angstrom
  (default ``10.0``)
* ``conv_use_gate`` (bool): if ``true``, use a PaiNN-style scalar gate in
  conv branches (default ``true``)

Example:

.. code-block:: yaml

   model:
     model_type: e3ee
     out_size: auto
     max_z: 100
     atom_emb_dim: 64
     atom_hidden_dim: 128
     atom_layers: 3
     local_cutoff: 6.0
     rbf_dim: 32
     energy_rbf_dim: 48
     scatter_dim: 128
     latent_dim: 128
     head_hidden_dim: 128
     e3nn_irreps: "64x0e + 32x1o + 16x2e"
     e3nn_irreps_message: "16x0e + 8x1o + 4x2e"
     e3nn_lmax: 2
     out_mlp_layers: 3
     use_invariant_branch: true
     use_attention_branch: true
     use_equivariant_branch: true
     use_eq_attention_branch: true
     use_conv_branch: false
     use_eq_conv_branch: false
     use_path_branch: false
     fusion_mode: cat
     residual_scale_init: 0.1
     attention_heads: 4
     attention_rbf_dim: 16
     attention_lmax: 2
     attention_irreps: "32x0e + 16x1o + 8x2e"
     att_cutoff: 10.0
     conv_use_gate: true

API Reference: :class:`E3EE <xanesnet.models.e3ee.e3ee.E3EE>`

EnvEmbed
--------

The environmental Embedding Network (EnvEmbed) is 
an absorber-centric environment embedding architecture to 
encode local atomic structure into a latent representation 
and predict Gaussian-transformed spectral coefficients.
It takes a :class:`EnvEmbed Data <xanesnet.datasets.torch.envembed.EnvEmbedData>` as input
and predicts its corresponding spectrum.

The model consists of two main components: 
a SoftRadialShellsEncoder and a Grouped Residual Head. 
The encoder performs soft radial binning of neighboring atoms 
around a central absorber atom using ``n_shells`` learnable distance 
shells within a cutoff radius defined by ``max_radius_angs``, 
with the Gaussian shell widths initialised by ``init_width``. 
If ``use_gating`` is ``true``, the shell
summary is also modulated by Fourier distance features.
The aggregated shell features are fused with the absorber 
representation and projected into a latent space. 
This latent embedding is then passed to the Grouped Residual Head, 
which applies a stack of Pre-LayerNorm residual feed-forward blocks. 
Each block consists of a normalization layer followed by a two-layer 
MLP with hidden dimension ``head_hidden``, GELU activation and dropout. 
Finally, multiple grouped linear heads generate structured coefficient 
outputs from the shared latent representation.


Parameters:

* ``model_type`` (str): must be ``envembed``
* ``in_size`` (int or ``auto``): descriptor feature dimension. ``auto``
  computes it from the descriptor at training time
* ``kgroups`` (list of int, or ``auto``): number of spectral-basis
  coefficients per width group. ``auto`` computes it from the dataset
  spectral basis at training time
* ``n_shells`` (int): number of learnable radial shells (default ``4``)
* ``max_radius_angs`` (float): radial cutoff in Angstrom (default
  ``7.0``)
* ``init_width`` (float): initial Gaussian shell width in Angstrom
  (default ``0.8``)
* ``use_gating`` (bool): if ``true``, modulate the shell summary with
  Fourier distance features (default ``true``)
* ``head_hidden`` (int): hidden dimension of the residual blocks in the
  coefficient head (default ``256``)
* ``head_depth`` (int): number of residual Pre-LN blocks in the
  coefficient head (default ``3``)
* ``dropout`` (float): dropout rate applied in the coefficient head,
  from ``0`` to ``1`` (default ``0.1``)

Example:

.. code-block:: yaml

   model:
     model_type: envembed
     in_size: auto
     kgroups: auto
     n_shells: 4
     max_radius_angs: 7.0
     init_width: 0.8
     use_gating: true
     head_hidden: 256
     head_depth: 3
     dropout: 0.1

API Reference: :class:`EnvEmbed <xanesnet.models.envembed.envembed.EnvEmbed>`

MH_MLP
------

A Multi-head MLP (MH_MLP) shares one MLP backbone across several prediction heads.
It takes a :class:`Multihead Data <xanesnet.datasets.torch.multihead.MultiheadData>` as input
and predicts one spectrum per head.

The shared backbone is the same as :class:`MLP <xanesnet.models.mlp.mlp.MLP>`:
a customizable sequence of hidden layers. Each hidden layer is composed of a
linear (dense) layer, a dropout layer for regularization, and a non-linear
activation function. The size of the first hidden layer is determined by the
``hidden_size`` parameter, and the size of subsequent hidden layers is
determined by the ``shrink_rate`` parameter, which multiplicatively reduces
the size of the previous layer.
For example, if ``hidden_size: 512`` and ``shrink_rate: 0.5``,
the shared hidden layers will have sizes ``512``, ``256``, and ``128``.

Each head is a individual MLP that maps the shared representation to one
spectrum. The architecture of each head is the same as :class:`MLP <xanesnet.models.mlp.mlp.MLP>`,
but with its own set of hidden layers.


Parameters:

* ``model_type`` (str): must be ``mh_mlp``
* ``in_size`` (int or ``auto``): length of the input features. ``auto``
  computes it from the descriptor at training time
* ``out_size`` (list of int, or ``auto``): length of the output features
  for each head. ``auto`` computes it from the targets at training time
* ``hidden_size`` (int): size of the first shared hidden layer (default
  ``226``)
* ``dropout`` (float): dropout rate applied after each hidden layer, from ``0`` to ``1`` (default ``0.1``)
* ``num_hidden_layers`` (int): Number of shared hidden layers excluding the heads (default ``3``).
* ``shrink_rate`` (float): Multiplicative factor applied to the shared layer width at each depth step (default ``0.5``).
* ``activation`` (str):  Name of the activation function for hidden layers
  (default ``prelu``). Allowed values: ``relu``, ``prelu``, ``tanh``,
  ``sigmoid``, ``elu``, ``leakyrelu``, ``selu``, ``silu``, ``gelu``
* ``head_hidden_size`` (int): size of the first hidden layer in each head (default ``226``)
* ``head_num_hidden_layers`` (int): number of hidden layers in each head excluding the output layer (default ``2``).
* ``head_shrink_rate`` (float): Multiplicative factor applied to each head's layer width at each depth step (default ``1.0``).

Example:

.. code-block:: yaml

   model:
     model_type: mh_mlp
     in_size: auto
     out_size: auto
     hidden_size: 512
     head_hidden_size: 512
     dropout: 0.1
     num_hidden_layers: 3
     head_num_hidden_layers: 2
     shrink_rate: 0.5
     head_shrink_rate: 1.0
     activation: prelu

API Reference: :class:`MultiHead_MLP <xanesnet.models.multihead.mh_mlp.MultiHead_MLP>`

MH_CNN
------

A Multi-head CNN (MH_CNN) shares a 1-D convolutional encoder across several
prediction heads.
It takes a :class:`Multihead Data <xanesnet.datasets.torch.multihead.MultiheadData>` as input
and predicts one spectrum per head.

The shared encoder is a sequence of convolutional layers. Each layer is
composed of a 1-D convolution, batch normalisation, a non-linear
activation function, and a dropout layer for regularization.
The first layer has ``out_channel`` output channels. Each later layer
multiplies the channel count by ``channel_mul``.
For example, if ``out_channel: 32``, ``channel_mul: 2``, and
``num_conv_layers: 3``, the layers have ``32``, ``64``, and ``128``
channels.

The flattened encoder output is passed to one MLP head per spectrum.
Head widths are set by ``head_hidden_size`` and ``head_shrink_rate``
in the same way as :class:`MLP <xanesnet.models.mlp.mlp.MLP>`. 

Parameters:

* ``model_type`` (str): must be ``mh_cnn``
* ``in_size`` (int or ``auto``): length of the input features. ``auto``
  computes it from the descriptor at training time
* ``out_size`` (list of int, or ``auto``): length of the output features
  for each head. ``auto`` computes it from the targets at training time
* ``hidden_size`` (int): size stored on the model signature (default
  ``512``)
* ``dropout`` (float): dropout rate applied after each convolutional layer, from ``0`` to ``1`` (default ``0.1``)
* ``num_conv_layers`` (int): Number of shared convolutional layers (default ``3``).
* ``activation`` (str):  Name of the activation function for convolutional and head layers
  (default ``prelu``). Allowed values: ``relu``, ``prelu``, ``tanh``,
  ``sigmoid``, ``elu``, ``leakyrelu``, ``selu``, ``silu``, ``gelu``
* ``out_channel`` (int): number of output channels in the first convolutional layer (default ``32``)
* ``channel_mul`` (int): Multiplicative factor applied to the channel count at each subsequent convolutional layer (default ``2``).
* ``kernel_size`` (int): size of the convolutional kernel (default ``3``)
* ``stride`` (int): stride of each convolutional layer (default ``2``)
* ``head_hidden_size`` (int): size of the first hidden layer in each head (default ``512``)
* ``head_num_hidden_layers`` (int): Number of hidden layers in each head excluding the output layer (default ``2``).
* ``head_shrink_rate`` (float): Multiplicative factor applied to each head's layer width at each depth step (default ``1.0``).

Example:

.. code-block:: yaml

   model:
     model_type: mh_cnn
     in_size: auto
     out_size: auto
     hidden_size: 512
     dropout: 0.1
     num_conv_layers: 3
     activation: prelu
     out_channel: 32
     channel_mul: 2
     kernel_size: 3
     stride: 2
     head_hidden_size: 512
     head_num_hidden_layers: 2
     head_shrink_rate: 1.0

API Reference: :class:`MultiHead_CNN <xanesnet.models.multihead.mh_cnn.MultiHead_CNN>`
