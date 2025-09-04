#!/bin/bash
# Shell script to test the theorem-compliant hyperparameter regime
# According to Theorem A.1:
# - K = Θ(D)
# - η₁ = Θ(1) (constant)
# - η₂ = Θ(D²)
# - η₃ = Θ(D⁻¹)
# - T = Ω(√(1+ε²)N⁴logN / (p_c²(1-p_b+Bp_b)²))

# Test with different values of D
for D in 32 64 128; do
    # Set hyperparameters according to theorem
    K=$D  # K = Θ(D)
    L=$D  # Keep L/K constant (ratio = 1)
    N=8   # Fixed N
    
    # Calculate learning rates based on D
    lr_stage1=1.0  # η₁ = Θ(1) - constant
    lr_stage2=$(python3 -c "print($D * $D)")  # η₂ = Θ(D²)
    lr_stage3=$(python3 -c "print(1.0 / $D)")  # η₃ = Θ(D⁻¹)
    
    # Calculate T based on theorem (simplified: using N⁴logN as baseline)
    # For p_B=0.5, p_C=0.5, B=2: denominator ≈ 0.25 * (0.5 + 2*0.5)² = 0.25 * 2.25 = 0.5625
    T_stage2=$(python3 -c "import math; print(int($N**4 * math.log($N) / 0.5625))")
    
    echo "Running experiment with D=$D, K=$K, L=$L"
    echo "Learning rates: lr1=$lr_stage1, lr2=$lr_stage2, lr3=$lr_stage3"
    echo "Training steps for stage 2: T=$T_stage2"
    
    python3 train_stages_new.py \
        0.5 0.5 \
        0 1 0 \
        $T_stage2 \
        $lr_stage1 $lr_stage2 $lr_stage3 \
        --D $D \
        --K $K \
        --L $L \
        --N $N \
        --lamb 1e-9
    
    echo "----------------------------------------"
done