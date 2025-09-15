#!/bin/bash --login
#SBATCH --job-name=Chisquare  # create a short name for your job
#SBATCH --nodes=1                # node count
#SBATCH --ntasks=1               # total number of tasks across all nodes
#SBATCH --gpus=h200:4            # number of gpus
#SBATCH --mem=10G                # memory per cpu
#SBATCH --time=02:00:00          # time
#SBATCH --mail-user=mondalra@msu.edu

module purge
module load Conda/3
module load CUDA/12.4.0

export PATH=$PATH:$HOME/anaconda3/bin
conda activate myenv1

cd '/mnt/home/mondalra/Anomaly_Detection/LRSTS/Varying_Duration/jobs_dur_10/'
python LRSTS_dur.py --iter_options "{\"rep\": 18}"
