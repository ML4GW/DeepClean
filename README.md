# DeepClean
Single source for training, performing inference with, then analyzing the performance of [DeepClean](https://arxiv.org/abs/2005.06534), a neural network for performing low-frequency noise subtraction in gravitational-wave observatories.


## Project organization
Project is divided into `libs`, modular source libraries for performing the relevant signal processing and deep learning tasks, and `projects`, pipelines for training and inference built on top of these libraries. The [`gw-iaas`](https://github.com/fastmachinelearning/gw-iaas) project is included as a submodule for building a production-ready inference pipeline.


## Installation
Individual libraries and projects will have their own installation steps, but the primary environment-management tool used is [Poetry](https://python-poetry.org/), which can be [installed](https://python-poetry.org/docs/master/#installing-with-the-official-installer) via

```console
curl -sSL https://install.python-poetry.org | python3 - --preview
```

Certain projects will require the use of LIGO Data Analysis System (LDAS) tools for reading and writing [gravitational wave frame files](https://dcc.ligo.org/T970130/public), as well as LIGO Network Data Service (NDS) libraries for remotely fetching data. These libraries are only installable via [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), so its recommended to install the base image here

```console
conda env create -f environment.yaml
```

Then clone it for installing poetry dependencies in other projects. For example, for installing `projects/training`, you might do something like

```console
conda env create -n deepclean-train --clone deepclean-base
conda activate deepclean-train
poetry install
```
 For details on individual libraries or projects, consult their documentation.
 
