# DL/ML solutions for phase transmission of MOFs

## Pipeline

- [Preprocessing](preprocessing/pteproc_model.py) - preprocessing pipeline for input of `Reduce` model
  - [Preprocessing QMOF](preprocessing/cleaning_qmof_data.ipynb)
  - [Preprocessing classification dataset](preprocessing/cleaning_qmof_data.ipynb)
- [Reduce](model/reduce_model.py) - Autoencoder and Variational Autoencoder models used for reducing number of features
  - [Reduce analysis](model/reduce_analysis.ipynb)
- [Classification model](model/classification_model.py) - model used for classification
- [Full pipeline](pipeline.ipynb)

for next workflow i use [this article](https://habr.com/ru/articles/106912/)
