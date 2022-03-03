# DeepClean
Single source for training, performing inference with, then analyzing the performance of [DeepClean](https://arxiv.org/abs/2005.06534), a neural network for performing low-frequency noise subtraction in gravitational-wave observatories.


## Project organization
Project is divided into `libs`, modular source libraries for performing the relevant signal processing and deep learning tasks, and `projects`, pipelines for training and inference built on top of these libraries. The [`gw-iaas`](https://github.com/fastmachinelearning/gw-iaas) project is included as a submodule for building a production-ready inference pipeline.


## Installation
Individual libraries and projects will have their own installation steps, but the primary environment-management tool used is [Poetry](https://python-poetry.org/), which can be [installed](https://python-poetry.org/docs/master/#installing-with-the-official-installer) via

```console
curl -sSL https://install.python-poetry.org | python3 - --preview
```

Certain projects will require the use of LIGO Data Analysis System (LDAS) tools for reading and writing [gravitational wave frame files](https://dcc.ligo.org/T970130/public), as well as LIGO Network Data Service (NDS) libraries for remotely fetching data. These libraries are only installable via [Conda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html), and are included in the base environment file [`environment.yaml`](./environment.yaml).

Once you have Poetry and Conda installed, you can install a command line utility into your base Conda environment for building and running the projects in this repo via

```console
poetry install
```

If you don't want to install development dependencies like `pre-commit`, you can instead run

```console
poetry install --without dev
```

 You should then be able to run

```console
deepclean -h
```

and see something like

```console
 usage: deepclean [-h] [--log-file LOG_FILE] [--verbose] {run,build} ...

positional arguments:
  {run,build}

optional arguments:
  -h, --help           show this help message and exit
  --log-file LOG_FILE  Path to write logs to
  --verbose            Log verbosity
```

It is _not_ recommended that you install this command-line utility inside a virtual environment, as this environment will lack the Poetry Python APIs that `deepclean` uses to build and execute commands in other virtual environments.

To build projects, you can run `deepclean build <project directory>`, which will build a virtual environment for that project and install all the necessary dependencies, e.g.

```console
deepclean build projects/sandbox/train
```

Consult the documentation of individual projects to see what commands are exposed in their virtual environments.
