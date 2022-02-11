#! /bin/bash -e

PROJECT_ROOT=$PWD
DEFAULT_DIR=$PROJECT_ROOT/results/$(git rev-parse --short=8 HEAD)
PROJECT_DIRECTORY=${PROJECT_DIRECTORY:-$DEFAULT_DIR}

get_cmd() {
    cmd="PROJECT_DIRECTORY=${PROJECT_DIRECTORY} $1 --typeo ..:$1"
    if [[ ! -z "$2" ]]; then cmd+=":$2"; fi
    echo "/bin/bash -c $cmd"
}

run() {
    pipeline=$1
    cd $PROJECT_ROOT/$pipeline

    if [[ -z "$2" ]]; then
        # only 1 argument: this is a poetry-based pipeline
        # that doesn't have a subcommand
        poetry run $(get_cmd $pipeline)
    elif [[ "$2" == "conda" ]]; then
        # 2nd argument indicates that this is a conda
        # pipeline, but there's no subcommand to run
        conda activate deepclean-${pipeline}
        $(get_cmd $pipeline)
    elif [[ -z "$3" ]]; then
        # there's no 3rd argument, and the 2nd argument
        # isn't conda, so this is a poetry-based pipeline
        # that has a subcommand
        poetry run $(get_cmd $pipeline $2)
    elif [[ "$3" == "conda" ]]; then
        # this has a subcommand and 3rd argument
        # indicates pipeline is conda-based
        conda activate deepclean-${pipeline}
        $(get_cmd $pipeline $2)
    else
        # there's a 3rd argument but it isn't conda,
        # so we don't know what to do with it
        echo "Couldn't understand 3rd argument $3"
        exit 1
}

run train autoencoder conda
run export autoencoder
run infer conda
run analyze conda
