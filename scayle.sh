#!/bin/bash -l

#1) specify the environment variables of the SLURM job -----------------------------------------------------------------------------------------------

#SBATCH -p genoa # partition to submit to
#SBATCH -q normal
#SBATCH -J v2real
#SBATCH -o /scratch/cidaut_tic_1/cidaut_tic_1_7/outputs/finalGAN/esrgan_%j.out
#SBATCH -t 64:00:00
#SBATCH --gpus=1
#SBATCH --cpus-per-gpu=8       # Procesadores por GPU

#3) Activate the conda environment. You must specify the path to the conda.sh script in your user directory. ------------------------------------------
source /home/cidaut_tic_1/COMUNES/miniconda3/etc/profile.d/conda.sh
conda activate venv_rain_cuda

# Check whether the environment has been successfully activated.
echo "Environment successfully activated. Executing the rest of the script..."

#5) run the script in all the nodes ---------------------------------------------------------------------------------------------------------------------------
set +e  

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan_stg1.yml --name esrgan_stg1 || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan_stg2.yml --name esrgan_stg2 || true

python3 run_sr.py --device 0 --test config/test.yml --config config/esrgan_stg3.yml --name esrgan_stg3 || true

conda deactivate

