#!/bin/bash
# Submission script for Lucia - CSLR (CTC) training.
#
#SBATCH --job-name=cslr_500
#SBATCH --time=48:00:00
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=32768
#SBATCH --partition=gpu
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/2-500/%j.out

module purge
module load EasyBuild/2025a
module load CUDA/12.8.0

source ~/miniconda3/etc/profile.d/conda.sh
conda activate slp
export LD_LIBRARY_PATH=$CONDA_PREFIX/lib:$LD_LIBRARY_PATH

# Should show conda's libstdc++, not /lib64/
ldconfig -p | grep libstdc++
strings $CONDA_PREFIX/lib/libstdc++.so.6 | grep GLIBCXX_3.4.29

which python
python --version

repo_dir="$HOME/repositories/sign-language-vqvae"
config_file="configs/cluster/cslr_500.json"

cd "$repo_dir"

nvidia-smi
echo "Config file: $config_file"
echo "Job start at $(date)"
python -m sl_vqvae.scripts.train_cslr --config "$config_file"
echo "Job end at $(date)"
