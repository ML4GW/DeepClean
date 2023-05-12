# DeepClean Production Microservice
This project builds a complete end-to-end application for iteratively training, validating, and deploying DeepClean for subtracting 60Hz noise from low-latency frames.
Each component functions as a standalone service with a dedicated environment and purpose.
At the moment, those environments are built locally using Conda, but in the near future will be fully containerized and deployed via Singularity.
The services contained here are:
- `trainer`: Trains a DeepClean model on live data collected from low latency frames at some fixed cadence. All strain data used for training must have the `DMT-ANALYSIS_READY` flag turned on.
- `exporter`: Flask application that takes newly trained versions of DeepClean, accelerates them using TensorRT, and adds them to a Triton model repository.
- `cleaner`: Responsible for sending inference requests to DeepClean to compute noise predictions and subtract them from low-latency strain.
- `monitor`: Monitors the cleaned data produced by the latest version of DeepClean in order to validate it and assess whether it's ready to move into production.

Additionally, this project contains a `utils` library for defining utility functions shared by the various projects, and leverages a Triton inference server instance to handle serving DeepClean for inference and moving new models into production.
Each project is discussed in more detail [below](#project-overviews).

## Build and run instructions
On a node with at least 2 enterprise-grade GPUs, begin by installing the `pinto` environment management utility by following the instructions [here](https://github.com/ML4GW/pinto).
If you chose to go with the virtual environment installation of `pinto`, be sure to activate that environment (`conda activate pinto`) before proceeding with the instructions below.

First, begin by using `pinto` to build all of the projects' corresponding environments

```
./build.sh
```

Then, launch a `tmux` session with panes executing each of the individual services by running

```
./run.sh
```

## Configuration
Each project reads its arguments from a shared configuration file [`pyproject.toml`](./pyproject.toml), which is parsed during command line execution by the locally-maintained `typeo` library.
The configuration is broken up into different tables for each project, denoted `[tool.typeo.scripts.<project name>]`, as well as a `[tool.typeo.base]` table for defining shared parameters.
For explanations of each argument, consult the corresponding project's documentation.

## Project overviews
