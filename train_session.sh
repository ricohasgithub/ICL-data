#!/bin/bash

# Define arrays of different parameter values
RUN_NAME="K_session" # Name of the session by independent variable
EPOCHS=(10 20 30)
BATCH_SIZES=(32 64 128)
LEARNING_RATES=(0.001 0.0005 0.0001)
MODEL_TYPES=("resnet" "cnn" "lstm")

NUM_MODELS=${#EPOCHS[@]}
OUTPUT_DIR="./sessions/$RUN_NAME"

mkdir -p $OUTPUT_DIR

# Loop over the indices of the arrays
for (( i=0; i<$NUM_MODELS; i++ )); do
    EPOCH=${EPOCHS[$i]}
    BATCH_SIZE=${BATCH_SIZES[$i]}
    LEARNING_RATE=${LEARNING_RATES[$i]}
    MODEL_TYPE=${MODEL_TYPES[$i]}
    
    echo "Training $MODEL_TYPE with epochs=$EPOCH, batch_size=$BATCH_SIZE, learning_rate=$LEARNING_RATE", run_name=$RUN_NAME
    python3 train.py --epochs $EPOCH --batch_size $BATCH_SIZE --learning_rate $LEARNING_RATE --model_type $MODEL_TYPE --run_name $RUN_NAME
done
