# The WorldStrat Software Package

This is the companion code repository for [the WorldStrat dataset](https://zenodo.org/record/6810792) and its article, used to generate the dataset and train several super-resolution benchmarks on it. The associated article and datasheet for dataset is [available on arXiv](https://arxiv.org/abs/2207.06418).

# Quick Start
1. Download and install Mambaforge ([Windows](https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Windows-x86_64.exe)/[Linux](https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-Linux-x86_64.sh)/[Mac OS X](https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-MacOSX-x86_64.sh)/[Mac OS X ARM](https://github.com/conda-forge/miniforge/releases/download/4.12.0-2/Mambaforge-4.12.0-2-MacOSX-arm64.sh)/[Other](https://github.com/conda-forge/miniforge/releases/tag/4.12.0-2))
2. Open a Miniforge prompt or [initialise Mambaforge in your terminal/shell](https://docs.conda.io/projects/conda/en/latest/dev-guide/deep-dives/activation.html#conda-initialization) (conda init).
3. Clone the repository: `git clone https://github.com/worldstrat/worldstrat`.
4. Install the environment: `mamba env create -n worldstrat --file environment.yml`.
5. Follow the instructions in the `Dataset Exploration` notebook using the `worldstrat` environment.
6. Alternatively (manual):
  1. [Download the dataset from Zenodo](https://zenodo.org/record/6810792), or [from Kaggle](https://www.kaggle.com/datasets/jucor1/worldstrat)
  2. Create an empty `dataset` folder in the repository root (`worldstrat/dataset`) and unpack the dataset there.
  3. Run the `Dataset Exploration` notebook, or any of the other notebooks, using the `worldstrat` environment.

## What is WorldStrat?

**Nearly 10,000 km² of free high-resolution satellite imagery** of unique locations which ensure stratified representation of all types of land-use across the world: from agriculture to ice caps, from forests to multiple urbanization densities.

![A mosaic showing randomly selected high-resolution imagery from the dataset.](https://i.imgur.com/cESfjpB.png)

Those locations are also enriched with typically under-represented locations in ML datasets: sites of humanitarian interest, illegal mining sites, and settlements of persons at risk.

Each high-resolution image (1.5 m/pixel) comes with multiple temporally-matched low-resolution images from the freely accessible lower-resolution Sentinel-2 satellites (10 m/pixel). 

We accompany this dataset with a paper, datasheet for datasets and an open-source Python package to: rebuild or extend the WorldStrat dataset, train and infer baseline algorithms, and learn with abundant tutorials, all compatible with the popular EO-learn toolbox.

![A world map showing the location of the dataset imagery with their source labels (ASMSpotter, Amnesty, UNHCR, Randomly Sampled/Landcover).](https://i.imgur.com/QLpnXE5.jpeg)

# Why make this?

We hope to foster broad-spectrum applications of ML to satellite imagery, and possibly develop the same power of analysis allowed by costly private high-resolution imagery from free public low-resolution Sentinel2 imagery. We illustrate this specific point by training and releasing several highly compute-efficient baselines on the task of Multi-Frame Super-Resolution. 

# Data versions and structure:
The main repository for this dataset is [Zenodo](https://zenodo.org/record/6810792).
It contains:
- [12-bit radiometry high-resolution images in their raw format](https://zenodo.org/record/6810792/files/hr_dataset_raw.tar.gz?download=1), downloaded directly from Airbus.
- [12-bit radiometry high-resolution images](https://zenodo.org/record/6810792/files/hr_dataset.tar.gz?download=1) [downloaded through and processed by SentinelHub](https://docs.sentinel-hub.com/api/latest/data/airbus/spot/).
- [16 temporally-matched low-resolution Sentinel-2 Level-2A revisits](https://zenodo.org/record/6810792/files/lr_dataset_l2a.tar.gz?download=1) for each high-resolution image.
- [16 temporally-matched low-resolution Sentinel-2 Level-1C revisits](https://zenodo.org/record/6810792/files/lr_dataset_l1c.tar.gz?download=1) for each high-resolution image.
- [The metadata about the dataset](https://zenodo.org/record/6810792/files/metadata.csv?download=1) (imaged locations coordinates, several classifications).
- [The train/val/test split](https://zenodo.org/record/6810792/files/stratified_train_val_test_split.csv?download=1) used to train the super-resolution benchmarks.
- [The scientific paper about the dataset and toolbox published and presented at NeurIPS 2022](https://zenodo.org/record/6810792/files/WorldStrat_article_and_datasheet.pdf?download=1).


Due to Kaggle's size limitation of ~107 GB, [we've uploaded what we call the "core dataset" there](https://www.kaggle.com/datasets/jucor1/worldstrat), which consists of:

- 12-bit radiometry high-resolution images, downloaded through SentinelHub's API.
- 8 temporally-matched low-resolution Sentinel-2 Level-2A revisits for each high-resolution image.

We used this core dataset to train the models we used as benchmarks in our paper, and which we distribute as pre-trained models.



# How can I use this?

We recommend starting with the downloading and unpacking the dataset, and using the `Dataset Exploration` notebook to explore the data.  
After that, you can also check out our source code which contains notebooks that demonstrate:

- Generating the dataset by randomly sampling the entire planet and stratifying the points using several datasets.
- Training a super-resolution model that generates high-resolution imagery using low-resolution Sentinel-2 imagery as input. 
- Running inference/generating free super-resolved high-resolution imagery using the aforementioned model.

![An image demonstrating the difference between a low-resolution image and it's super-resolved high-resolution image, generated using the pre-trained model.](https://i.imgur.com/aVL9Jy4.png)

# Licences 

- The high-resolution Airbus imagery is distributed, with authorization from Airbus, under Creative Commons Attribution-NonCommercial 4.0 International (CC BY-NC 4.0).
- The labels, Sentinel2 imagery, and trained weights are released under Creative Commons with Attribution 4.0 International (CC BY 4.0).
- This source code repository under 3-Clause BSD license.

# How to cite

If you use this package or the associated dataset, please kindly cite these following BibTeX entries:

```
@misc{cornebise_open_2022,
  title = {Open {{High-Resolution Satellite Imagery}}: {{The WorldStrat Dataset}} -- {{With Application}} to {{Super-Resolution}}},
  author = {Cornebise, Julien and Or{\v s}oli{\'c}, Ivan and Kalaitzis, Freddie},
  year = {2022},
  month = jul,
  number = {arXiv:2207.06418},
  eprint = {2207.06418},
  eprinttype = {arxiv},
  publisher = {{arXiv}},
  doi = {10.48550/arXiv.2207.06418},
  archiveprefix = {arXiv}
}

@article{cornebise_worldstrat_zenodo_2022,
  title = {The {{WorldStrat Dataset}}},
  author = {Cornebise, Julien and Orsolic, Ivan and Kalaitzis, Freddie},
  year = {2022},
  month = jul,
  journal = {Dataset on Zendodo},
  doi = {10.5281/zenodo.6810792}
}
```
