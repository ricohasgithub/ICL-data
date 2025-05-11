#!/bin/bash
##SBATCH --mem=128G
#SBATCH --job-name=icl_experiment
#SBATCH -t 120:00:00  # time requested in hour:minute:second
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=logs/launcher-%j.out
#SBATCH --error=logs/launcher-%j.err

# Fixed hyperparameters
p_B=0.75
p_C=0.75

# make sure output directories exist
mkdir -p sbatch_scripts
mkdir -p logs

# loop over all circuit/block combinations
for circuit_num in {0..3}; do
  for block0_num in {0..3}; do
    for block1_num in {0..3}; do

      job_script="sbatch_scripts/train_c${circuit_num}_b0${block0_num}_b1${block1_num}.sh"

      cat <<EOT > "$job_script"
#!/bin/bash
#SBATCH --mem=128G
#SBATCH --job-name=tr_c${circuit_num}_b0${block0_num}_b1${block1_num}
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=logs/c${circuit_num}_b0${block0_num}_b1${block1_num}-%j.out
#SBATCH --error=logs/c${circuit_num}_b0${block0_num}_b1${block1_num}-%j.err

python3 ./train.py $p_B $p_C $circuit_num $block0_num $block1_num
EOT

      # submit the generated job
      sbatch "$job_script"

    done
  done
done
