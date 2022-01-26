# DeepClean Projects
Here you can find individual pipelines for performing DeepClean-related tasks such as training, inference, and analysis, built on top of a shared set of libraries and using shared config options via the [`pyproject.toml`](./pyproject.toml) `[tool.typeo]` tables.

## [`training`](./training)
A pipeline for training DeepClean from local HDF5 files, falling back to archival NDS2 data if the data doesn't exist locally.

## [`export`](./export)
A pipeline for exporting a trained DeepClean instance to a Triton model repository for as-a-service inference.
