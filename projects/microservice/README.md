# DeepClean Production Microservice
This project builds a complete end-to-end application for iteratively training, validating, and deploying DeepClean for subtracting 60Hz noise from low-latency frames.
Each component functions as a standalone service with a dedicated environment and purpose.
At the moment, those environments are built locally using Conda, but in the near future will be fully containerized and deployed via Singularity.
The services contained here are:
- `trainer`: Trains a DeepClean model on live data collected from low latency frames at some fixed cadence. All strain data used for training must have the `DMT-ANALYSIS_READY` flag turned on.
- `exporter`: Flask application that takes newly trained versions of DeepClean, accelerates them using TensorRT, and adds them to a Triton model repository.
- `cleaner`: Responsible for sending inference requests to DeepClean to compute noise predictions and subtract them from low-latency strain.
- `monitor`: Monitors the cleaned data produced by the latest version of DeepClean in order to validate it and assess whether it's ready to move into production.

Additionally, this project contains a `utils` library for defining utility functions shared by the various projects, and leverages a [Triton inference server](https://docs.nvidia.com/deeplearning/triton-inference-server/user-guide/docs/index.html) instance to handle serving DeepClean for inference and moving new models into production.
Each project is discussed in more detail [below](#project-overviews).

## Build and run instructions
> Note: future deployment strategy will leverage Singularity containers pushed to GitHub registry automatically via CI, and made available on LDG via the OSG container repository.
This strategy will be implemented once the current PR is pushed to main and the container builds can be tested.

On a node with at least 2 enterprise-grade GPUs, begin by installing the `pinto` environment management utility by following the instructions [here](https://github.com/ML4GW/pinto).
If you chose to go with the virtual environment installation of `pinto`, be sure to activate that environment (`conda activate pinto`) before proceeding with the instructions below.

First, begin by using `pinto` to build all of the projects' corresponding environments

```bash
./build.sh
```

Next, define the required environment variables in a local `.env` file pointing to the relevant I/O directories.
The minimum required environment variables are

```bash
# Note that you shouldn't define the $PROJECT
# variable here, it will be defined when you
# launch the service and filled in here to
# given individual runs their own directories
PROJECT_DIR=${HOME}/deepclean/${PROJECT}  # path to save logs, cleaned frames, versioned models, etc.
DATA_DIR=/dev/shm/  # path to input data
IFO=L1  # interferometer being run on
```

Then, launch a `tmux` session with panes executing each of the individual services by running

```bash
./run.sh first-pipeline-run
```

where `first-pipeline-run` is used to set `PROJECT` environment variable used to name individual pipeline runs.
Alternatively, you can run each one of the commands implemented in [`run.sh`](./run.sh) inside a pre-configured `tmux` session.

## Configuration
Each project reads its arguments from a shared configuration file [`pyproject.toml`](./pyproject.toml), which is parsed during command line execution by the locally-maintained `typeo` library.
The configuration is broken up into different tables for each project, denoted `[tool.typeo.scripts.<project name>]`, as well as a `[tool.typeo.base]` table for defining shared parameters.
For explanations of each argument, consult the corresponding project's documentation. Instructions for running each project individually can be found by inspecting `run.sh`(./run.sh).

## Project overviews
### `trainer`
Iteratively trains a new version of DeepClean on overlapping stretches of data.
For every training run after the first, the weights from the previously trained version of the model are used to initialize the weights for the next run, which are then optimized with a reduced learning rate.

While the model trains, its next training set is collected from frames in real time.
As noted above, all training data must have the `DMT-ANALYSIS_READY` flag set.
If at any point during data collection, a frame has any samples for which this flag is not set, the frame and existing training data are dropped.
Data collection resumes again once this flag is set again for a sufficiently long segment of frames.

If any dropped frames are encountered, the previous frame of data is tapered using a Hann window and a frame of all zeros is appended to the data stream.
Once a non-dropped frame re-enters the stream, its reentry is tapered using a Hann window.
If too many consecutive frames are dropped, the current training set is dropped and data collection resumes once frames become available again.

For details on the training process, consult the [associated publication](https://dcc.ligo.org/LIGO-P2300153). The relevant arguments can be found by running (in the associated environment)

```bash
train -h
```

The parameter values used in production are available in the [`pyproject.toml`](./pyproject.toml).

### `exporter`
This service runs a Flask app that manages the repository of models used to executed DeepClean in production, including the multiple versions of the DeepClean model itself.
DeepClean versions are associated with the timestamps used to train them via metadata in the DeepClean model config.

At any given moment, two versions of DeepClean are producing predictions.
The **production** model has had its predictions validated by the `monitor` service and has the quality of its ASD ratio actively monitored to ensure the final data product has no artifacts.
The **canary** model usually represents the most recently trained version of DeepClean which is waiting to have its performance validated by the `monitor` service. The `exporter` service
- creates the server-side pipeline to produce both sets of predictions in parallel
- exposes an endpoint for the `trainer` service to notify it about newly trained versions of DeepClean for export
- exposes an endpoint for the `monitor` service to notify that a recently trained "canary" version of DeepClean has passed validation and should be moved into production.

This service can only be run via the `pyproject.toml` config, which should be consulted for information about the relevant arguments.

### `cleaner`
Service responsible for producing a 300s long rotating carousel of frames containing cleaned strain data.
Reads strain and auxiliary channels from low-latency frames and sends them to a running Triton instance serving up both the production and canary versions of DeepClean and uses their predictions to produce cleaned frames.
Note that this service is completely agnostic to which _specific_ versions of the model are being served up by Triton. It just makes requests to a pipeline containing "production" and "canary" models, and its up to the `exporter` service which actual versions of the model weights these correspond to.

#### Data Products
Each frame produced by DeepClean will have 3 channels associated with it, each sampled at **4096 Hz**. The channels are:
- `<IFO>:GDSD-CALIB_STRAIN_DEEPCLEAN_PRODUCTION_OUT_DQ` - The official data product of DeepClean. The data contained in this channel will have been cleaned by a validated and stable version of DeepClean, and will have the content of its data monitored _prior_ to writing.
If the data is found to cause increases in the average ASD ratio over the relevant frequency band over an 8 second period including the current frame, the timeseries of noise predictions that is typically subtracted will be tapered out using a Hann window and only reintroduced once that ASD ratio falls under 1 again.
Until that point, this channel will just contain the raw strain data.
- `<IFO>:GDS-CALIB_STRAIN_DEEPCLEAN_PRODUCTION` - Similar to the `OUT_DQ` channel, but with subtraction _always_ turned on.
Intended mostly for analysis purposes.
- `<IFO>:GDS-CALIB_STRAIN_DEEPCLEAN_CANARY` - Frames cleaned using predictions from the canary model, used by the `monitor` service to validate the performance of the model.
Since the model used to clean the data in this channel will have been trained on more recent data, it may have superior performance to the data in the `PRODUCTION` channel, but we make no guarantees about the quality of the data in this channel.

#### Missing data
Unlike the `trainer`, this will produce predictions _regardless_ of the current data quality status, leaving that to downstream pipelines to decide what data to use.
However, it handles dropped frames the same way the `trainer` project does: tapering out the timeseries before the dropped frame using a Hann window, appending a frame of all 0s to each channel, then tapering back in the next non-dropped frame.

#### Sources of latency
Signal processing techniques and data safety measures necessitate some procedures which prevent DeepClean from being able to produce predictions on the most recent low-latency frame.
Moreover, because frames are only made available in discreet 1 second increments, each piece of new data required means waiting a full second for the entire next frame to become available.
This introduces some latency into the DeepClean pipeline that has nothing to do with its compute requirements, which are comparably light.
Those latency sources are:
- Preprocessing
    - Data resampling: downsampling the data from 16384 to 4096 Hz introduces artifacts at the edges of the downsampled timeseries.
    To combat this, we maintain a 3-second buffer used to downsample, and take the middle second as the next frame to send through DeepClean.
    This introduces 1 second of latency waiting for future data to fill out the buffer.
    - Tapering dropped frames: We need to check if the next frame has been dropped before appending to our buffer, because if it has then we need to taper the end of our current timeseries to avoid introducing noise artifacts when we append all 0s.
    This introduces another second of latency waiting for the next frame to become available.
- Postprocessing
    - Averaging predictions: DeepClean produces 512 1-second long noise prediction timeseries each second. That means that each of these predictions carries a lot of overlap, and predictions made farther from the edge of the 1-second long window tend to be better as a consequence of the way DeepClean is trained.
    This means we can improve our performance by averaging over the overlapping portions of these predictions.
    This means predictions at the end of a frame can't be made until they've been averaged with predictions on that same time segment from a timeseries which contains data _after_ the frame as well.
    This introduces another 1 second of latency waiting for the next frame to be preprocessed and sent through DeepClean.
    - Filter padding: Once a frame of (averaged) noise predictions has been produced by DeepClean, we need to bandpass filter it to the target frequency range before subtracting it from the strain channel.
    Like the downsampling, this introduces artifacts at the edges of the timeseries being filtered.
    So we wait for another 0.5s worth of averaged predictions to become available to pad our timeseries and ensure that we have well behaved data.
    Because the averaging mentioned above only takes place over 0.5s, this actually ends up not introducing any additional latency, but is worth mentioning since it would under different circumstances (e.g. if low-latency frames were written at a faster cadence).
- Data safety
    - To ensure that DeepClean doesn't introduce additional noise artifacts into the data, we maintain an 8s buffer of cleaned data, of which the frame to be written is the second-to-last.
    Before writing the `OUT_DQ` channel, we measure the ASD ratio over this buffer on the target frequency range.
    If the average of that ratio is greater than 1, we elect not to write our noise-subtracted data and instead taper the noise prediction from the second-to-last frame.
    Until the ASD ratio returns below 1, we continue writing raw strain data to the `OUT_DQ` channel (the regular `PRODUCTION` channel, on the other hand, will always contain the noise-subtracted data used to compute this ASD ratio).
    This introduces another second of latency waiting for the next frame to be cleaned.

In total, our data processing strategy introduces 4s of latency, while our compute itself takes on the order of 200ms.
This is obviously less than ideal, but worthwhile to ensure that the data produced by DeepClean does not degrade the quality data used by downstream applications.

### `monitor`
The monitoring service serves two primary functions
- Monitors the quality of data produced by the canary model to validate whether it was trained sufficiently well to be put in production.
If the ASD ratio of the canary cleaned data with the raw strain data is less than 1 over a 100 second period, it is deemed ready to go into production and a request is made to the `exporter` service to increment the DeepClean version corresponding to the production model.
- Ensures that the carousel of frames produced by the `cleaner` service never exceeds 300, and periodically saves larger stretches (O(1000s)) of cleaned and raw data to cold storage for offline analysis and testing.
