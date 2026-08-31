#!/bin/bash
# Submission script for Lucia - WP2_replication (BEST) training.
#
# train.py's __main__ picks the stage to run (tokenizer / pretrain / finetune)
# via commented-out lines, exactly as it does locally. Edit WP2_replication/train.py
# to select the stage BEFORE submitting this job.
#
#SBATCH --job-name=wp2_best
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
#SBATCH --output=./out/train/%j.out

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

export WP2_DATA_ROOT="/gpfs/scratch/acad/lsfb/datasets/lsfb-cont"
export WP2_OUTPUT_DIR="/gpfs/scratch/acad/lsfb/outputs/wp2-best-replication/checkpoints"

cd "$repo_dir/WP2_replication"

nvidia-smi
echo "Data root: $WP2_DATA_ROOT"
echo "Output dir: $WP2_OUTPUT_DIR"
echo "Job start at $(date)"
python train.py
echo "Job end at $(date)"
