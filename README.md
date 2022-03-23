# DeepClean
Single source for training, performing inference with, then analyzing the performance of [DeepClean](https://arxiv.org/abs/2005.06534), a neural network for performing low-frequency noise subtraction in gravitational-wave observatories.


## Project organization
Project is divided into `libs`, modular source libraries for performing the relevant signal processing and deep learning tasks, and `projects`, pipelines for training and inference built on top of these libraries. The [`gw-iaas`](https://github.com/fastmachinelearning/gw-iaas) project is included as a submodule for building a production-ready inference pipeline.


## Installation
### Setting up the repository
Before you do anything, be sure that the `gw-iaas` submodule has been initialized after cloning this repo
```
git submodule update
git submodule init
```

### Environmnet setup
#### 1. The Easy Way - `pinto`
The simplest way to interact with the code in this respository is to install the ML4GW [Pinto command line utility](https://ml4gw.github.io/Pinto/), which contains all the same prerequisites (namely Conda and Poetry) that this repo does.
The only difference is that rather than having to keep track of which projects require Conda and which only need Poetry separately, `pinto` exposes commands which build and execute scripts inside of virtual environments automatically, dynamically detecting how and where to install each project.
For more information, consult the Pinto documentation linked to above.

#### 2. The Hard Way - Conda + Poetry
Otherwise, make sure you have Conda and Poetry installed in the manner outlined in Pinto's documentation.
Then create the base Conda environment on which all projects are based

```console
conda env create -f environment.yaml
```

Projects that requires Conda will have a `poetry.toml` file in them containing in part

```toml
[virtualenvs]
create = false
```

For these projects, you can build the necessary virtual environment by running 

```console
conda create -n <environment name> --clone deepclean-base
```

then using Poetry to install additional dependencies (called from the project's root directory, not the repository's)

```console
poetry install
```

Otherwise, you should just need to run `poetry install` from the project's directory, and Poetry will take care of creating a virtual environment automatically.

### Running projects
Consult each project's documentation for additional installation steps and available commands. If you installed the `pinto` utility, commands can be run
in each project's virtual environment by

```console
pinto run path/to/project my-command --arg 1
```

Otherwise, for Conda projects, you'll have to activate the appropriate Conda environment first then execute the code inside of it.
For Poetry environments, it should be enough to run `poetry run my-command --arg 1` _from the projects directory_ (one downside of Poetry is that everything has to be done locally).
