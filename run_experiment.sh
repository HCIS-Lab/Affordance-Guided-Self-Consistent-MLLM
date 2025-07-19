#!/bin/bash

# Default values for optional variables
LOG_ROOT="experiment_log"  # Default path to experiment logs
CONFIG_FILE="src/config/final_task/all.yaml"
NOW_TIME=$(LC_TIME=en_US.utf8 date +%Y%m%d_%H%M%S)
DEFAULT_TASKS=("distance" "spatial_relationship" "general_easy" "amount_ambiguity" "obstacles")
TASKS=""
# Function to display usage
usage() {
    echo "Usage: $0 -n <PIPELINE_NAME> [ -e <EXP_ID>] [-c <CONFIG_FILE] [-l <LOG_ROOT>] [-t <TASKS>]"
    echo "  -n PIPELINE_NAME: Required. Type of test to run."
    echo "  -e EXP_ID: Optional. Experiment ID. (default: None, will use current time as ID)"
    echo "  -c CONFIG_FILE: Optional. Path to your config file. (default: $CONFIG_FILE)"
    echo "  -l LOG_ROOT: Optional. Path to experiment logs (default: $LOG_ROOT)."
    echo "  -t TASKS: Optional. Comma-separated list of tasks to run (default: all tasks)."
    echo "  -h: Show this help message."
    exit 1
}
# Parse flags
while getopts "e:c:n:l:t:h" opt; do
    case $opt in
        e) EXP_ID=$OPTARG ;;
        c) CONFIG_FILE=$OPTARG ;;
        n) PIPELINE_NAME=$OPTARG ;;
        l) LOG_ROOT=$OPTARG ;;  
        t) TASKS=$OPTARG ;;  
        h) usage ;;
        *) usage ;;
    esac
done

# Convert TASKS string to array if provided via command line
if [ -n "$TASKS" ]; then
    # Convert comma-separated string to array
    IFS=',' read -ra TASK_ARRAY <<< "$TASKS"
else
    # Use default tasks
    TASK_ARRAY=("${DEFAULT_TASKS[@]}")
fi

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
export PIPELINE_NAME=$PIPELINE_NAME
export CONFIG_FILE=$CONFIG_FILE

echo "Running experiment $EXP_ID with config file $CONFIG_FILE"
sleep 1

python concat_config.py
CONFIGURATIONS=$(yq 'to_entries | map(.key as $parent | .value | to_entries | map([$parent, .key])) | flatten' $CONFIG_FILE | sed '/^#/d; s/ #.*//' | sed 's/- //' | tr -d '",[]')
config_array=($CONFIGURATIONS)

for ((i = 0; i < ${#config_array[@]}; i+=2)); do
    trap "echo 'Keyboard Interupt.'; pkill -P $$; exit 130" SIGINT
    export TASK_TYPE=${config_array[i]}
    export ENV_IDX=${config_array[i+1]}
    
    # Check if this task should be run (check against TASK_ARRAY)
    should_run_task=false
    for task in "${TASK_ARRAY[@]}"; do
        if [[ "$task" == "$TASK_TYPE" ]]; then
            should_run_task=true
            break
        fi
    done
    
    if [ "$should_run_task" = false ]; then
        continue
    fi
    echo "Running task $TASK_TYPE with environment $ENV_IDX"
    python experiment.py
done

# echo "Reached maximum number of trials ($MAX_TRIALS). Exiting..."

echo "Convert codec of video"
python convert_codec.py $RESULT_DIR
exit 1
