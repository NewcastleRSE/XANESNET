<table align="center">
<tr><td align="center" width="10000">

<img src = "docs/source/images/xanesnet_graphic.png" width = "380">

# <strong> X A N E S N E T </strong>

<p>
    <a href="http://penfoldgroup.co.uk">Penfold Group </a> @ <a href="https://ncl.ac.uk">Newcastle University </a>
</p>

<p>
    <a href="https://xanesnet.readthedocs.io">User Manual</a> • <a href="#setup">Setup</a> • <a href="#getting">Quickstart</a> • <a href="#publications">Publications</a>
</p>

</td></tr></table>

#

We think that the theoretical simulation of X-ray spectroscopy (XS) should be fast, affordable, and accessible to all researchers. 

The popularity of XS is on a steep upward trajectory globally, driven by advances at, and widening access to, high-brilliance light sources such as synchrotrons and X-ray free-electron lasers (XFELs). However, the high resolution of modern X-ray spectra, coupled with ever-increasing data acquisition rates, brings into focus the challenge of accurately and cost-effectively analyzing these data. Decoding the dense information content of modern X-ray spectra demands detailed theoretical calculations that are capable of capturing satisfactorily the complexity of the underlying physics but that are - at the same time - fast, affordable, and accessible enough to appeal to researchers. 

This is a tall order - but we're using deep neural networks to make this a reality. 

Our XANESNET software address two fundamental challenges: the so-called forward (property/structure-to-spectrum) and reverse (spectrum-to-property/structure) mapping problems. The forward mapping appraoch is similar to the appraoch used by computational researchers in the sense that an input structure is used to generate a spectral observable. In this area the objective of XANESNET is to supplement and support analysis provided by first principles quantum mechnanical simulations. The reverse mapping problem is perhaps the more natural of the two, as it has a clear connection to the problem that X-ray spectroscopists face day-to-day in their work: how can a measurement/observable be interpreted? Here we are seeking to provide methodologies in allow the direct extraction of properties from a recorded spectrum. 

XANESNET is under continuous development, so feel free to flag up any issues/make pull requests - we appreciate your input!

The original version of XANESNET, which was implemented using Keras, can be obtained from <a href="https://gitlab.com/team-xnet/xanesnet_keras">here</a>. The <a href="https://xanesnet.readthedocs.io">XANESNET User Manual</a> has more information about the code and its uses.

## Features

* GPLv3 licensed open-source distribution
* Automated data processing: Fourier transform, Gaussian transform
* Feature extraction: wACSF, RDC, pDOS, MACE
* Neural network architecture: MLP, CNN, GNN, LSTM, Autoencoder, Autoencoder-GAN, Multihead, Transformer, EnvEmbed
* Learning scheme: standard, K-fold, ensemble learning, bootstrapping
* Experiment tracking and visualisation: MLFlow, TensorBoard
* Learning rate scheduler
* Custom workflow components (datasets, models, descriptors, etc.) defined via input file
* Easy to extend with new components 
* Web interface


## Setup

The quickest way to get started with XANESNET is to clone this repository:


<!---
```
git clone https://gitlab.com/team-xnet/xanesnet.git 
```
--->

```
git clone https://gitlab.com/team-xnet/xanesnet.git
```

The repository contains all source files, along with example input files and datasets.

Complete training sets for X-ray absorption and emission of molecules containing first row transition metals can be obtained using:

```
git clone https://gitlab.com/team-xnet/training-sets.git
```

Now you're good to go!

## Getting Started

The code has been designed to support python 3.10 and above. 
Dependencies and version requirements can be installed using:

```
python -m pip install .
```

### Training 

To train a model, use the following command:  

```python3 -m xanesnet.cli --mode MODE --in_file <path/to/file.yaml> --save```

where
- `--mode` specifies the training mode.
- `--in_file` specifies the path to the input file for training.
- `--save` specifies whether to save the trained model to disk.


The implemented training modes include:  
- `train_xyz`: uses featurised structures as input data and XANES spectra as the target.
- `train_xanes`: uses XANES spectra as input data and the featurised structures as the target.
- `train_all`: trains both featurised structures and XANES spectra simultaneously (Only available for the AEGAN model type).

Replace <path/to/file.yaml> with the path to your YAML input file.
Examples of commented input files for training and hyperparameter 
configuration can be found in the 'inputs/' directory.

Below is an example command for training a model using the MLP architecture, with featurised structures as input data:  

```python3 -m xanesnet.cli --mode train_xyz --in_file inputs/in_mlp.yaml --save```

The resulting trained model and its metadata will be saved in the 'models/' directory. 

### Prediction

To use a previously trained model for predictions, use the following command:

```python3 -m xanesnet.cli --mode MODE --in_model <path/to/model> --in_file <path/to/file.yaml>```

where
- `--mode` specifies the prediction mode.
- `--in_model` specifies a directory containing a pre-trained model and its metadata.
- `--in_file` specifies the path to the input file for prediction.

The implemented prediction modes include:  
- `predict_xanes`: predicts a XANES spectrum from a featurised structural input.
- `predict_xyz`: predicts featurised structures from an input XANES spectrum.
- `predict_all`: simultaneous prediction of both featurised structures and XANES spectra from corresponding inputs with reconstruction of inputs. 


As an example, the following command predicts XANES spectra using the MLP model trained previously:

```python3 -m xanesnet.cli --mode predict_xanes --in_model models/mlp_std_xyz_001 --in_file inputs/in_predict.yaml```


## Contact

### Project Team

<a href="https://ncl.ac.uk/nes/people/profile/tompenfold.html">Prof. Thomas Penfold </a>, Newcastle University, (tom.penfold@newcastle.ac.uk)\
<a href="https://www.ncl.ac.uk/nes/people/profile/thomaspope2.html">Dr. Thomas Pope </a>, Newcastle University (thomas.pope2@newcastle.ac.uk)\
<a href="https://pure.york.ac.uk/portal/en/persons/conor-rankine">Dr. Conor Rankine </a>, York University (conor.rankine@york.ac.uk)


### RSE Contact
<a href="https://rse.ncldata.dev/team/bowen-li">Dr. Bowen Li </a>, Newcastle University (bowen.li2@newcastle.ac.uk)

## License

This project is licensed under the GPL-3.0 License - see the LICENSE.md file for details.

## Publications

### The Program:
*[A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra](https://doi.org/10.1021/acs.jpca.0c03723)* - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

*[Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network](https://doi.org/10.1063/5.0087255)* - C. D. Rankine, and T. J. Penfold, *J. Chem. Phys.*, 2022, **156**, 164102.
 
#### Extension to X-ray Emission:
*[A deep neural network for valence-to-core X-ray emission spectroscopy](https://doi.org/10.1080/00268976.2022.2123406)* - T. J. Penfold, and C. D. Rankine, *Mol. Phys.*, 2022, e2123406.

#### The Applications:
*[On the Analysis of X-ray Absorption Spectra for Polyoxometallates](https://doi.org/10.1016/j.cplett.2021.138893)* - E. Falbo, C. D. Rankine, and T. J. Penfold, *Chem. Phys. Lett.*, 2021, **780**, 138893.

*[Enhancing the Anaysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump-X-ray Probe Experiments](https://doi.org/10.1039/D0CP06244H)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

#### Miscellaneous:
*[The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy](https://doi.org/10.3390/molecules25112715)* - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.
