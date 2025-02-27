#!/bin/bash
#SBATCH --mem=128G
#SBATCH --job-name=icl_experiment_launcher
#SBATCH -t 120:00:00
#SBATCH --partition=compsci-gpu
#SBATCH --output=icl_out/launcher-%j.out
#SBATCH --error=icl_out/launcher-%j.err

# Loop over combinations of p_B and p_C
for p_B in $(seq 0.0 0.05 1.0); do
    for p_C in $(seq 0.0 0.05 1.0); do
        # Generate a unique job script for each combination
        job_script="icl_job_pB_${p_B}_pC_${p_C}.sh"

        # Write the job script
        cat <<EOT > $job_script
#!/bin/bash
#SBATCH --mem=128G
#SBATCH --job-name=icl_pB_${p_B}_pC_${p_C}
#SBATCH -t 120:00:00
#SBATCH --gres=gpu:1
#SBATCH --partition=compsci-gpu
#SBATCH --output=icl_out/pB_${p_B}_pC_${p_C}-%j.out
#SBATCH --error=icl_out/pB_${p_B}_pC_${p_C}-%j.err

# Run the Python script with the current p_B and p_C values
python3 ./train.py ${p_B} ${p_C}
EOT

        # Submit the job script
        sbatch $job_script
    done
done
