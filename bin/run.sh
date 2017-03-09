#!/bin/bash

script="$0"
FOLDER="$(pwd)/$(dirname $script)"

source $FOLDER/utils.sh
PROJECT_ROOT="$(abspath $FOLDER/..)"
echo "project root folder $PROJECT_ROOT"

echo "build docker image"
/bin/bash $FOLDER/build.sh

##### VOLUMES #####

# folder containing model
MODEL_DIR=$PROJECT_ROOT/output
echo "Writing outputs to $MODEL_DIR"

##### RUN #####
echo "Starting container..."

docker run --rm \
           --name intent \
           -it \
           -p 5000:5000 \
           -v $MODEL_DIR:/output \
           dominicbreuker/spacy_intent:latest \
           /bin/sh -c "python /intent/app.py"
