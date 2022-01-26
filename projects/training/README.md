# DeepClean training
Trains DeepClean using either local hdf5 files or by fetching data from LIGO's Network Data Service (NDS) archive.

Almost all of the actual training is handled by the [`deepclean.trainer`](../../libs/trainer) library, so consult the code there to get a sense for what the actual training loop looks like. The code here is primarily interested in the logic for getting the specified data and preparing it into numpy arrays, as well in setting up logging.

## Installation
As mentioned in the [root README](../../README.md), you can install this project by first creating then cloning the base deepclean environment. From the root directory of this repo, this looks like

```console
conda env create -f environment.yaml
conda env create -n deepclean-train --clone deepclean-base
```

Then you can install the necessary additional libraries by running in this directory

```console
conda activate deepclean-train
poetry install
```

## Running
To get a sense for what command line arguments are available, you can run (from this directory as well):

```console
train -h
```

To use the default values saved in the `tool.typeo` section of the [pyproject.toml](./pyproject.toml), run

```console
train --typeo ::autoencoder
```

Where here `autoencoder` specifies the network architecture that we'd like to train with (it's the only one for now).
