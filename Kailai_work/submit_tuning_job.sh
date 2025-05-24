#!/bin/bash
#SBATCH --job-name=dataset_tuning
#SBATCH --output=dataset_tuning_%j.out
#SBATCH --error=dataset_tuning_%j.err
#SBATCH --time=4:00:00
#SBATCH --partition=normal
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# Load necessary modules
module load python/3.9

# Define important paths
SCRIPT_DIR="/home/kw395/debug_project/scripts"
NOTEBOOK_DIR="/home/kw395/debug_project/notebooks"
PARAM_FILE="/home/kw395/debug_project/Parameter_Values.json"
OUTPUT_DIR="/home/kw395/debug_project/results"

# Create output directory if it doesn't exist
mkdir -p ${OUTPUT_DIR}

# Run the parameter tuning script with dataset-specific optimization flags
python ${SCRIPT_DIR}/parameter_tuning_v2.py \
    --notebooks_dir ${NOTEBOOK_DIR} \
    --parameter_file ${PARAM_FILE} \
    --output_dir ${OUTPUT_DIR} \
    --dataset_specific_optimizations \
    --fill_missing_metrics \
    --optimize_fillna \
    --max_concurrent_notebooks 4

# Generate a summary report
echo "Parameter tuning completed. Results saved to ${OUTPUT_DIR}" 