#!/bin/bash --login
#SBATCH --job-name=Chisquare
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gpus=h200:2
#SBATCH --exclude=neh-000
#SBATCH --mem=10G
#SBATCH --time=02:00:00
#SBATCH --mail-user=mondalra@msu.edu
module purge
module load Conda/3
module load CUDA/12.4.0
export PATH=$PATH:$HOME/anaconda3/bin
conda activate myenv1
cd '/mnt/home/mondalra/Anomaly_Detection/LRSTSS/Varying_SNR/jobs_snr_20/'
python LRSTSS_snr.py --iter_options "{\"rep\": 15}"
