# Analyze DeepClean Outputs
Produce an HTML document summarizing the performance of a DeepClean network using the outputs from the other modules in this workflow. This includes plots of the ASD before and after cleaning, a plot of the ratio between these zoomed in on the frequency band of interest, a plot of the training and validation loss curves, and a text box with tabs for logs and configs from the run.

NOTE: Won't work for arbitrary runs of DeepClean, designed to work specifically with this workflow.


## Installation
Assuming you chose [installation path #1](../../../README.md#environment-setup) when setting this repo up and have the `pinto` command line utility available, you can install this project simply via

```console
pinto build .
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
To get a sense for what command line arguments are available, you can either run:

```console
pinto run . analyze -h
```

if you have the `pinto` command line utility installed, or you can `conda activate deepclean-analyze` and just run `analyze -h`.
Either way, you should see something like

```console
usage: main [-h] --raw-data-dir RAW_DATA_DIR --clean-data-dir CLEAN_DATA_DIR --output-directory OUTPUT_DIRECTORY
            --channels CHANNELS [CHANNELS ...] --sample-rate SAMPLE_RATE --fftlength FFTLENGTH [--freq-low FREQ_LOW]
            [--freq-high FREQ_HIGH] [--overlap OVERLAP]

    Build an HTML document analyzing a set of gravitational
    wave frame files cleaned using DeepClean. This includes
    plots of both the cleaned and uncleaned ASDs, as well as
    of the ratio of these ASDs plotted over the frequency
    range of interest. Included above these plots are the
    training and validation loss curves from DeepClean
    training as well as a box including any relevant logs
    or configs used to generate this run of DeepClean.

optional arguments:
  -h, --help            show this help message and exit
  --raw-data-dir RAW_DATA_DIR
                        Directory containing the raw frame files containing the strain channel DeepClean was used to
                        clean (default: None)
  --clean-data-dir CLEAN_DATA_DIR
                        Directory containing the frame files produced by DeepClean with the cleaned strain channel
                        (whose name should match the raw strain channel) (default: None)
  --output-directory OUTPUT_DIRECTORY
                        Directory to which HTML document should be written as `analysis.html`. Should also include any
                        log files (ending `.log`) that are desired to be included in the plot. (default: None)
  --channels CHANNELS [CHANNELS ...]
                        A list of channel names used by DeepClean, with the strain channel first, or the path to a
                        text file containing this list separated by newlines (default: None)
  --sample-rate SAMPLE_RATE
                        Rate at which the input data to DeepClean was sampled, in Hz (default: None)
  --fftlength FFTLENGTH
  --freq-low FREQ_LOW   The low end of the frequency range of interest for plotting the ASD ratio (default: None)
  --freq-high FREQ_HIGH
                        The high end of the frequency range of interest for plotting the ASD ratio (default: None)
  --overlap OVERLAP     The amount of overlap, in seconds, between successive FFT windows for the ASD computation. If
                        left as `None`, this will default to `fftlength / 2`. (default: None)
```
