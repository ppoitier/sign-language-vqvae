#!/bin/bash
# Submission script for Lucia - single-dictionary transformer VQ-VAE training.
#
#SBATCH --job-name=vqvae_transformer
#SBATCH --time=24:00:00
#
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --gres="gpu:1"
#SBATCH --mem-per-cpu=16384
#SBATCH --partition=gpu
#
#SBATCH --mail-user=pierre.poitier@unamur.be
#SBATCH --mail-type=ALL
#
#SBATCH --account=lsfb
#
#SBATCH --output=./out/2-transformer/%j.out

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
config_file="configs/cluster/transformer_vqvae.json"

cd "$repo_dir"

nvidia-smi
echo "Config file: $config_file"
echo "Job start at $(date)"
python -m sl_vqvae.scripts.train --config "$config_file"
echo "Job end at $(date)"
