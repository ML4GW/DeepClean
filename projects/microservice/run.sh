#! /bin/bash -e

PROJECT=$1

prun() {
    echo "PROJECT=$PROJECT pinto -p $1 run -e ../.env"
}
tmux new-session -s deepclean-production -d "CUDA_VISIBLE_DEVICES=0 $(prun exporter) flask --app=exporter run"

while [[ -z $(curl -s localhost:5000/alive) ]]; do
    echo "Waiting for export service to come online"
    sleep 2
done

tmux split-window -v "CUDA_VISIBLE_DEVICES=0 $(prun trainer) train --typeo pyproject.toml script=train architecture=autoencoder"
tmux split-window -h "
    CUDA_VISIBLE_DEVICES=1 singularity exec \
        --nv /cvmfs/singularity.opensciencegrid.org/ml4gw/hermes/tritonserver:22.12 \
            /opt/tritonserver/bin/tritonserver \
            --model-repository ~/deepclean/results/microservice/$PROJECT/model_repo \
            --model-control-mode poll \
            --repository-poll-secs 10 \
"
tmux split-window -v "$(prun cleaner) clean --typeo pyproject.toml script=clean"
tmux split-window -v "$(prun monitor) monitor --typeo pyproject.toml script=monitor"
