#!/bin/bash -x

PROJECT=$1

prun() {
    echo "PROJECT=$PROJECT pinto -p $1 run -e ../.env"
}
tmux new-session -s deepclean-production -d "CUDA_VISIBLE_DEVICES=0 $(prun exporter) flask --app=exporter run"
tmux split-window -v "CUDA_VISIBLE_DEVICES=0 $(prun trainer) train --typeo pyproject.toml script=train architecture=autoencoder"
tmux split-window -h "CUDA_VISIBLE_DEVICES=1 singularity exec --nv /cvmfs/singularity.opensciencegrid.org/ml4gw/tritonserver:22.12 "
