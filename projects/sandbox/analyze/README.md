# Analyze DeepClean Outputs
Produce an HTML document summarizing the performance of a DeepClean network using the outputs from the other modules in this workflow. This includes plots of the ASD before and after cleaning, a plot of the ratio between these zoomed in on the frequency band of interest, a plot of the training and validation loss curves, and a text box with tabs for logs and configs from the run.

NOTE: Won't work for arbitrary runs of DeepClean, designed to work specifically with this workflow.


## Installation
If you've followed the steps outlined in the root [README](../../../README.md) for installing the DeepClean command line utility, you can install this project simply via

```console
deepclean build .
```

from this directory. Otherwise, you can install this project by first creating then cloning the base deepclean environment. From the root directory of this repo, this looks like

```console
conda env create -f environment.yaml
conda env create -n deepclean-analyze --clone deepclean-base
```

Then you can install the necessary additional libraries by running in this directory

```console
conda activate deepclean-analyze
poetry install
```

## Available commands
### `analyze`
To get a sense for what command line arguments are available, you can run (from the `deepclean-analyze` conda environment: `conda activate deepclean-analyze`):

```console
analyze -h
```

and should see something like

```console
```
