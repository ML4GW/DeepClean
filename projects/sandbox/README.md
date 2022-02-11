# DeepClean Sandbox Projects
Here you can find components for performing vanilla DeepClean pipeline tasks such as training, inference, and analysis, built on top of a shared set of libraries and using shared config options via the [`pyproject.toml`](./pyproject.toml) `[tool.typeo]` tables.

## [`training`](./training)
Train DeepClean using local HDF5 files, falling back to archival NDS2 data if the data doesn't exist locally.

## [`export`](./export)
Exporting a trained DeepClean instance to a Triton model repository for as-a-service inference.

## [`inference`](./inference)
Use a Triton server instance hosting DeepClean to clean a directory of strain GWF files in-memory, using witnesses from another directory of GWF files, and write the results to this pipeline's project directory.

## [`analysis`](./analysis)
Compute and plot the ASDs of the raw and cleaned frames, as well as the ratio between their ASDs in the target frequency band. Write these to an HTML file in the pipeline's project directory.
