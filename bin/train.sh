#!/bin/bash

script="$0"
FOLDER="$(pwd)/$(dirname $script)"

source $FOLDER/utils.sh
PROJECT_ROOT="$(abspath $FOLDER/..)"
echo "project root folder $PROJECT_ROOT"

echo "build docker image"
/bin/bash $FOLDER/build.sh

##### VOLUMES #####

# folder containing data
DATA_DIR=$PROJECT_ROOT/data
echo "Using data in $DATA_DIR"

# folder containing output
OUTPUT_DIR=$PROJECT_ROOT/output
echo "Writing outputs to $OUTPUT_DIR"

##### RUN #####
echo "Starting container..."

docker run --rm \
           -it \
           -v $DATA_DIR:/data \
           -v $OUTPUT_DIR:/output \
           dominicbreuker/spacy_intent:latest \
           /bin/sh -c "python /intent/train.py"
