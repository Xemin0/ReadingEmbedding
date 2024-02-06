#!/bin/bash                                                                                               

# Job Name
#SBATCH -J RE

# Time Requested (Not yet tested)
#SBATCH -t 0:30:00

# Number of Cores (max 4)
#SBATCH -c 2

# Single Node (Not yet optimized for Distributed Computing)
#SBATCH -N 1

# Request GPU partition and Access (max 2)
#SBATCH -p gpu --gres=gpu:1

# Request Memory (Not yet tested)
#SBATCH --mem=15G

# Outputs
#SBATCH -e ./scratch/RE.err
#SBATCH -o ./scratch/RE.out

########### END OF SLURM COMMANDS ##############

# Show CPU infos
lscpu

# Show GPU infos
nvidia-smi

# Force Printing Out `stdout`
export PYTHONBUFFERED=TRUE


# Run the Python file with arguments
python3 trainREmodel.py
