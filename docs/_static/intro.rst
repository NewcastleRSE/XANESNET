Introduction
==================

---------------------
XANESNET Overview
---------------------

XANESNET is an open-source toolbox for the rapid and
automated analysis and prediction of X-ray spectroscopy data.
The toolbox provides solution for both forward and reverse mapping problems,
linking material properties or structures with their corresponding
X-ray Absorption Near Edge Structure (XANES) spectra.

The forward mapping approach is similar to the approach used by
computational researchers in the sense that an input structure is used to generate a spectral observable.
In this area the objective of XANESNET is to supplement and support analysis provided by first principles
quantum mechanical simulations. The reverse mapping problem, on the other hand,
focuses on interpreting the material property from XANES spectra.
This is perhaps the more natural of the two,
as it has a clear connection to the problem that X-ray spectroscopists face day-to-day in their work:
how can a measurement/observable be interpreted?

XANESNET approaches this problem by employing machine learning models,
including a variety of deep neural network architectures to perform inverse analysis.
By training these models on large datasets,
XANESNET can accurately predict the characteristics of unknown samples.
The choice of neural network architectures enables users to select the most
appropriate model based on their specific requirements.


---------------------
XANESNET features
---------------------

* GPLv3 licensed open-source distribution
* Automated data processing: data augmentation, Fourier transform
* Feature extraction: wACSF, SOAP, MBTR, LMBTR, RDC, LMBTR, pDOS, MSR
* Neural network architecture: MLP, CNN, LSTM, Autoencoder, Autoencoder-GAN
* Learning scheme: K-fold, ensemble, bootstrap
* Hyperparameter optimisation
* Learning rate scheduler
* Layer freezing
* Run from an input file
* Easy to extend with new model and descriptor
* Web interface

--------------------------
XANESNET development team
--------------------------

XANESNET is developed by the
`Penfold Group <http://penfoldgroup.co.uk>`_ and
the `Research Software Engineering (RSE) team <https://rse.ncldata.dev/>`_
at `Newcastle University <https://ncl.ac.uk>`_.

| Project team:
| `Prof. Thomas Penfold <https://www.ncl.ac.uk/nes/people/profile/tompenfold.html>`_ (tom.penfold@newcastle.ac.uk)
| `Dr. Conor Rankine <https://www.york.ac.uk/chemistry/people/conor-rankine/>`_ (conor.rankine@york.ac.uk)

| RSE team:
| `Dr. Bowen Li <https://rse.ncldata.dev/team/bowen-li>`_ (bowen.li2@newcastle.ac.uk)
| `Alex Surtee <https://rse.ncldata.dev/team/alex-surtees>`_ (alex.surtees@newcastle.ac.uk)

| Former RSEs:
| Dr. Nik Khadijah Nik Aznan
| Dr. Kathryn Garside

------------
Publications
------------

**XANESNET Code**:

`A Deep Neural Network for the Rapid Prediction of X-ray Absorption Spectra <https://doi.org/10.1021/acs.jpca.0c03723>`_ - C. D. Rankine, M. M. M. Madkhali, and T. J. Penfold, *J. Phys. Chem. A*, 2020, **124**, 4263-4270.

`Accurate, affordable, and generalizable machine learning simulations of transition metal x-ray absorption spectra using the XANESNET deep neural network <https://doi.org/10.1063/5.0087255>`_ - C. D. Rankine, and T. J. Penfold, *J. Chem. Phys.*, 2022, **156**, 164102.

**Extension to X-ray Emission**:

`A deep neural network for valence-to-core X-ray emission spectroscopy <https://doi.org/10.1080/00268976.2022.2123406>`_ - T. J. Penfold, and C. D. Rankine, *Mol. Phys.*, 2022, e2123406.


**The Applications**:

`On the Analysis of X-ray Absorption Spectra for Polyoxometallates <https://doi.org/10.1016/j.cplett.2021.138893>`_ - E. Falbo, C. D. Rankine, and T. J. Penfold, *Chem. Phys. Lett.*, 2021, **780**, 138893.

`Enhancing the Anaysis of Disorder in X-ray Absorption Spectra: Application of Deep Neural Networks to T-Jump-X-ray Probe Experiments <https://doi.org/10.1039/D0CP06244H>`_ - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Phys. Chem. Chem. Phys.*, 2021, **23**, 9259-9269.

**Miscellaneous**:

`The Role of Structural Representation in the Performance of a Deep Neural Network for X-ray Spectroscopy <https://doi.org/10.3390/molecules25112715>`_ - M. M. M. Madkhali, C. D. Rankine, and T. J. Penfold, *Molecules*, 2020, **25**, 2715.



---------------
Citing XANESNET
---------------

.. code-block::

    @software{xanesnet,
      author = {Penfold Group, Newcastle University},
      title = {XANESNET},
      url = {https://gitlab.com/team-xnet/xanesnet},
      date = {2023},
    }
