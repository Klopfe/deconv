#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH -c 6
#SBATCH --time=0-24:00:00
#SBATCH -p batch
# module use $LOCAL_MODULES
# module load tools/EasyBuild
source /opt/apps/resif/iris/2019b/broadwell/software/Anaconda3/2020.02/etc/profile.d/conda.sh
module load lang/Anaconda3
module load lang/R/3.6.2-foss-2019b-bare
conda activate deconvolution_env
python3 heatmap_noise.py