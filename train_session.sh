#!/bin/bash

# Define arrays of different parameter values
RUN_NAME="K_session" # Name of the session by independent variable
EPOCHS_=(10 20 30)
K_=(512 1024 2048)
L_=(32 32 32)
S_=(10000 10000 10000)
N_=(8 8 8)
NMAX_=(32 32 32)
EPS_=(0.1 0.1 0.1)

ALPHA_=(0.0 0.0 0.0)
B_=(1 1 1)

PB_=(1.0 1.0 1.0)
PC_=(0.8 0.8 0.8)

BATCHSIZE_=(128 128 128)
NOREPEATS_=(false false false)


NUM_MODELS=${#EPOCHS[@]}
OUTPUT_DIR="./sessions/$RUN_NAME"

mkdir -p $OUTPUT_DIR

# Loop over the indices of the arrays
for (( i=0; i<$NUM_MODELS; i++ )); do
    EPOCHS=${EPOCHS_[$i]}
    K=${K_[$i]}
    L=${L_[$i]}
    S=${S_[$i]}
    N=${N_[$i]}
    NMAX=${NMAX_[$i]}
    EPS=${EPS_[$i]}
    ALPHA=${ALPHA_[$i]}
    B=${B_[$i]}
    PB=${PB_[$i]}
    PC=${PC_[$i]}
    BATCHSIZE=${BATCHSIZE_[$i]}
    NOREPEATS=${NOREPEATS_[$i]}
    
    echo "Training model with EPOCHS=$EPOCHS, K=$K, L=$L, S=$S, N=$N, NMAX=$NMAX, EPS=$EPS, ALPHA=$ALPHA, B=$B, PB=$PB, PC=$PC, BATCHSIZE=$BATCHSIZE, NOREPEATS=$NOREPEATS", RUN_NAME=$RUN_NAME
    python3 train.py --epochs $EPOCHS --K $K --L $L --S $S --N $N --Nmax $NMAX --eps $EPS --alpha $ALPHA --B $B --p_B $PB --p_C $PC --batchsize $BATCHSIZE --no_repeats $NOREPEATS --run_name $RUN_NAME
done
