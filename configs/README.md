User Input
===


## Data Params

```
data_params: True
augment:
    augment_type: random_noise
    augment_mult: 5
    augment_params: 
    normal_mean: 0
    normal_sd: 0.1
```

There are currently two options for data `augment_type`: `random_noise` and `random_combination`.

The random noise option adds Normally distributed noise to existing data. The mean and standard deviation of this distribution are provided as `normal_mean` and `normal_sd` in the `augment_params` dictionary.

The random combination option creates a new augmented data as the pointwise mean of two random data. This option requires no additional arguments.

The amount by which to increase the dataset via augmentation is given as an integer in `augment_mult`.

## Hyperparams

```
hyperparams: 
  model: mlp
  batch_size: 64
  n_hl: 2
  hl_ini_dim: 512
  hl_shrink: 0.5
  activation: prelu
  loss:
    loss_fn: mse
    loss_args: null
  lr: 0.00001
  dropout: 0.2
  weight_init_seed: 2023
  kernel_init: xavier_uniform
  bias_init: zeros
```

### Activation Functions

| `activation` | Description |
| --- | ----------- |
| `relu`      | Rectified Linear Unit               |
| `prelu`     | Parameterised Rectified Linear Unit |
| `tanh`      | Tanh                                |
| `sigmoid`   | Sigmoid                             |
| `elu`       | Exponential Linear Unit             |
| `leakyrelu` | Leaky Rectified Linear Unit         |
| `selu`      | Scaled Exponential Linear Unit      |

**Example:** `"activation": "prelu"`


### Loss Functions 

| `loss_fn`   | Description | `loss_args` |
| ---      | ----------- | ---|
| `mse`    | Mean Squared Error                   | n/a  |
| `bce`    | Binary Cross Entropy Loss with Logits| n/a |
| `emd`    | Earth Mover or Wasserstein Distance  | n/a |
| `cosine` | Cosine Similarity                    | n/a |
| `l1`     | L1 Loss                              | n/a |
| `wcc`    | Weighted Cross Correlation           | Gaussian HWHM :`float` |

**Example:** ```"loss": {"loss_fn" : "wcc", "loss_args" : 10}```

### Model Weight Initialisation


| Option   | Description | 
| ---      | ----------- | 
| `uniform`         | Uniformly distributed between (0,1) |
| `normal`          | Normally distributed wiht mean 0 and sd 1|
| `xavier_uniform`  | Uniform distribution according to [Bengio et. al (2010)](http://proceedings.mlr.press/v9/glorot10a)|
| `xavier_normal`   | Normal distribution according to [Bengio et. al (2010)](http://proceedings.mlr.press/v9/glorot10a)|
| `kaiming_uniform` | Uniform distribution according to [He et. al (2015)](http://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)|
| `kaiming_normal`  | Normal distribution according to [He et. al (2015)](http://openaccess.thecvf.com/content_iccv_2015/html/He_Delving_Deep_into_ICCV_2015_paper.html)|
| `zeros`           | Scalar Value 0|
| `ones`            | Scalar Value 1|


**Example:** 
```
"weight_init_seed":2023,
"kernel_init": "xavier_uniform",
"bias_init": "zeros"
```

### Learning Rate Scheduler
[Pytorch lr_scheduler](https://pytorch.org/docs/stable/optim.html)

| `type`   | Description | `params` |
| ---      | ------ | ---|
| `step`    | Decays the learning rate of each parameter group by gamma every step_size epochs                  | step_size: `int` <br />gamma:`float` |
| `multistep`    | Decays the learning rate of each parameter group by gamma once the number of epoch reaches one of the milestones. | milestones:`list` <br /> gamma: `float` |
| `exponential`    | Decays the learning rate of each parameter group by gamma every epoch  | gamma: `float` |
| `linear` | Decays the learning rate of each parameter group by linearly changing small multiplicative factor until the number of epoch reaches a pre-defined milestone: total_iters.                    | start_factor :`float` <br /> end_factor:`float` <br /> total_iters:`int`   |
| `constant`     | Decays the learning rate of each parameter group by a small constant factor until the number of epoch reaches a pre-defined milestone: total_iters.                             | factor: `float` <br /> total_iter:`int` |

**Example:** 

```
lr_scheduler:  
  scheduler: True
  scheduler_type: step
  scheduler_param:
    step_size: 50
    gamma: 0.1```