# DeepClean Sandbox Projects
Here you can find components for performing vanilla DeepClean workflow tasks such as training, inference, and analysis, built on top of a shared set of libraries and using shared config options via the [`pyproject.toml`](./pyproject.toml) `[tool.typeo]` tables.

## Components
### [`training`](./training)
Train DeepClean using local HDF5 files, falling back to archival NDS2 data if the data doesn't exist locally.

### [`export`](./export)
Exporting a trained DeepClean instance to a Triton model repository for as-a-service inference.

### [`inference`](./inference)
Use a Triton server instance hosting DeepClean to clean a directory of strain GWF files in-memory, using witnesses from another directory of GWF files, and write the results to this pipeline's project directory.

### [`analysis`](./analysis)
Compute and plot the ASDs of the raw and cleaned frames, as well as the ratio between their ASDs in the target frequency band. Write these plot, the training curves, and the logs from each job to an HTML file in the pipeline's project directory.

## Running
If you followed the `deepclean` command line interface installation instructions outlined in the root [README](../../README.md), you can build each component's environment individually by running  `deepclean build <path to component>`. Consult each component's documentation for the commands and options they expose.

Running the workflow end-to-end requires exporting three environment variables (with apologies for the inconsistent naming conventions for now)

```console
PROJECT_DIRECTORY="..."  # the directory to write workflow outputs to
DATA_DIR="..."  # the path to a directory of 1s .gwf witness/strain files
MODEL_REPOSITORY="..."  # the directory to serve models from
```

Note that `DATA_DIR` is expected to have the structure

```console
$DATA_DIR/
    | lldetchar/
        | H1/
            | <witness frame file 1>.gwf
            | <witness frame file 2>.gwf
            ...
    | llhoft/
        | H1/
            | <strain frame file 1>.gwf
            | <strain frame file 2>.gwf
            ...
```

Once these have been exported to your environment, you can run the workflow end-to-end simply by executing

```console
deepclean run .
```

from this directory. This will take care of building all the necessary virtual environment and executing the component scripts from inside of them.
