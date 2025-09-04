#!/bin/bash
# Shell script to test non-ICL regime based on data distribution parameters
# These configurations should NOT achieve good ICL performance due to data properties
# Keeping 0 < p_B, p_C < 1 as required by theorem

echo "=== Testing non-ICL regimes (data distribution variations) ==="

# Base hyperparameters following theorem
D=64
K=$D
L=$D
N=8
lr_stage1=1.0
lr_stage2=$(python3 -c "print($D * $D)")
lr_stage3=$(python3 -c "print(1.0 / $D)")
T_stage2=$(python3 -c "import math; print(int($N**4 * math.log($N) / 0.5625))")

# Experiment 1: Very low p_B and p_C (minimal burstiness and consistency)
echo "Experiment 1: p_B = 0.01, p_C = 0.01 (minimal ICL structure)"
python3 train_stages_new.py \
    0.01 0.01 \
    0 1 0 \
    $T_stage2 \
    $lr_stage1 $lr_stage2 $lr_stage3 \
    --D $D \
    --K $K \
    --L $L \
    --N $N \
    --B 2

echo "----------------------------------------"

# Experiment 2: Low p_C (poor label consistency)
echo "Experiment 2: p_B = 0.5, p_C = 0.05 (poor label consistency)"
python3 train_stages_new.py \
    0.5 0.05 \
    0 1 0 \
    $T_stage2 \
    $lr_stage1 $lr_stage2 $lr_stage3 \
    --D $D \
    --K $K \
    --L $L \
    --N $N \
    --B 2

echo "----------------------------------------"

# Experiment 3: Low p_B (poor class burstiness)
echo "Experiment 3: p_B = 0.05, p_C = 0.5 (poor class burstiness)"
python3 train_stages_new.py \
    0.05 0.5 \
    0 1 0 \
    $T_stage2 \
    $lr_stage1 $lr_stage2 $lr_stage3 \
    --D $D \
    --K $K \
    --L $L \
    --N $N \
    --B 2

echo "----------------------------------------"

# Experiment 4: High noise with moderate p_B, p_C
echo "Experiment 4: p_B = 0.3, p_C = 0.3 with high noise epsilon = 0.3"
python3 train_stages_new.py \
    0.3 0.3 \
    0 1 0 \
    $T_stage2 \
    $lr_stage1 $lr_stage2 $lr_stage3 \
    --D $D \
    --K $K \
    --L $L \
    --N $N \
    --B 2 \
    --eps 0.3

echo "----------------------------------------"

# Experiment 5: Very high p_B, very low p_C (strong burstiness, weak consistency)
echo "Experiment 5: p_B = 0.95, p_C = 0.05 (strong burstiness, weak consistency)"
python3 train_stages_new.py \
    0.95 0.05 \
    0 1 0 \
    $T_stage2 \
    $lr_stage1 $lr_stage2 $lr_stage3 \
    --D $D \
    --K $K \
    --L $L \
    --N $N \
    --B 3

echo "----------------------------------------"

# Experiment 6: Very low p_B, very high p_C (weak burstiness, strong consistency)
echo "Experiment 6: p_B = 0.05, p_C = 0.95 (weak burstiness, strong consistency)"
python3 train_stages_new.py \
    0.05 0.95 \
    0 1 0 \
    $T_stage2 \
    $lr_stage1 $lr_stage2 $lr_stage3 \
    --D $D \
    --K $K \
    --L $L \
    --N $N \
    --B 2

echo "----------------------------------------"

# Experiment 7: Moderate noise with low p_B and p_C
echo "Experiment 7: p_B = 0.1, p_C = 0.1 with epsilon = 0.1"
python3 train_stages_new.py \
    0.1 0.1 \
    0 1 0 \
    $T_stage2 \
    $lr_stage1 $lr_stage2 $lr_stage3 \
    --D $D \
    --K $K \
    --L $L \
    --N $N \
    --B 2 \
    --eps 0.1

echo "=== Completed non-ICL experiments (data distribution) ==="