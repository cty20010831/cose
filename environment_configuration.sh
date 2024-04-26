#!/bin/bash

# This shell script is used to configure the environment for this project.
# It includes activating the virtual environment, updating PYTHONPATH and setting environment variables.

# Activate the virtual environment
source ../.venv/bin/activate
echo "Activated virtual environment venv_cose."

export PROJECT_CODE_PATH="/Users/samcong/Library/CloudStorage/OneDrive-TheUniversityofChicago/Thesis/cose"
export PYTHONPATH="${PYTHONPATH}:${PROJECT_CODE_PATH}"

export COSE_DATA_DIR="${PROJECT_CODE_PATH}/data_dir"
export COSE_LOG_DIR="${PROJECT_CODE_PATH}/log_dir"
export COSE_EVAL_DIR="${PROJECT_CODE_PATH}/eval_dir"
export GDRIVE_API_KEY=""

echo "Environment variables set:"
echo "PYTHONPATH=$PYTHONPATH"
echo "COSE_DATA_DIR=$COSE_DATA_DIR"
echo "COSE_LOG_DIR=$COSE_LOG_DIR"
echo "COSE_EVAL_DIR=$COSE_EVAL_DIR"
echo "GDRIVE_API_KEY=$GDRIVE_API_KEY"

# Run the code in the terminal 
# chmod +x environment_configuration.sh
# source environment_configuration.sh 
# conda deactivate