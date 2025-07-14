# Bayesian_DL_Notebooks
This repo contains of notebooks that include deep learning based Bayesian statistics workflows. Notebooks consists of data generation, model definition, training, and postprocessing.

Notebooks in both folders consists workflows with a conditional generative model that approximate distribution of a hidden variable from observations

First folder "1_JAX_examples_from_PyTorch" consists of a notebook that contains Variational Autoencoder (VAE) implementation in PyTorch, and in JAX (Flax) with comparable postprocessing results/plots. In addition to VAE model, in one notebook VAE is replaced with a Normalizing Flow architecture (Neural Spline Flow).

Second folder "2_Bayesflow_example_for_Spatial_Model" includes a notebook that trains a model which estimates distribution of an unobserved variable from observations in a grid. Generative process is a log-Gaussian-Cox Process, which is common choice for geostatistical models. It follows a method named as 'Amortized Bayesian Inference', and is using CNN as a summary network.


**Note:** All notebooks in this repository are tested in a Google Colab environment with GPU support.
