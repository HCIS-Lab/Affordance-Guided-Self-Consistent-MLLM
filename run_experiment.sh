#!/bin/bash

# Default values for optional variables
LOG_ROOT="experiment_log"  # Default path to experiment logs
MAX_TRIALS=5  # Default number of trials
CHECK_INTERVAL=5  # Time interval between checks (in seconds)
NOW_TIME=$(LC_TIME=en_US.utf8 date +%Y%m%d_%H%M%S)
# Function to display usage
usage() {
    echo "Usage: $0 -n <PIPELINE_NAME> [ -e <EXP_ID>] [-c <CONFIG_FILE] [-l <LOG_ROOT>] [-t <MAX_TRIALS>]"
    echo "  -e EXP_ID: Required. Experiment ID."
    echo "  -c CONFIG_FILE: Required. Path to your config file."
    echo "  -n PIPELINE_NAME: Required. Type of test to run."
    echo "  -l LOG_ROOT: Optional. Path to experiment logs (default: $LOG_ROOT)."
    echo "  -t MAX_TRIALS: Optional. Number of trials to perform (default: $MAX_TRIALS)."
    echo "  -h: Show this help message."
    exit 1
}

# Parse flags
while getopts "e:c:n:l:t:h" opt; do
    case $opt in
        e) EXP_ID=$OPTARG ;;
        c) CONFIG_FILE=$OPTARG ;;
        n) PIPELINE_NAME=$OPTARG ;;
        l) LOG_ROOT=$OPTARG ;;  # Override default LOG_ROOT if provided
        t) MAX_TRIALS=$OPTARG ;;  # Override default MAX_TRIALS if provided
        h) usage ;;
        *) usage ;;
    esac
done

# Ensure required flags are provided
if [ -z "$PIPELINE_NAME" ]; then
    echo "Error: -n (PIPELINE_NAME) is required."
    usage
fi

# Path to results folder
if [ -z "$EXP_ID" ]; then
    RESULT_DIR="${LOG_ROOT}/${PIPELINE_NAME}_${NOW_TIME}"
else
    RESULT_DIR="${LOG_ROOT}/${PIPELINE_NAME}_${NOW_TIME}_${EXP_ID}"
fi

echo "Results will be saved in $RESULT_DIR"

# Export environment variables for other scripts
export RESULT_DIR=$RESULT_DIR
export CONFIG_FILE=${CONFIG_FILE:-src/config/final_task/all.yaml}
export PIPELINE_NAME=$PIPELINE_NAME

echo "Running experiment $EXP_ID with config file $CONFIG_FILE"
sleep 1

python concat_config.py
CONFIGURATIONS=$(yq 'to_entries | .[:] | map(.key as $parent | .value | to_entries | .[:] | map([$parent, .key])) | flatten' $CONFIG_FILE | sed '/^#/d; s/ #.*//' | sed 's/- //')
config_array=($CONFIGURATIONS)

for ((i = 0; i < ${#config_array[@]}; i+=2)); do
    export TASK_TYPE=${config_array[i]}
    export ENV_IDX=${config_array[i+1]}
    echo "Running task $TASK_TYPE with environment $ENV_IDX"
    python experiment.py
done

# echo "Reached maximum number of trials ($MAX_TRIALS). Exiting..."

echo "Convert codec of video"
python convert_codec.py $RESULT_DIR
exit 1
