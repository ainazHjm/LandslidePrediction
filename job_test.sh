#!/bin/bash
#SBATCH --account=def-davpoole
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=11000M
#SBATCH --time=0-01:00            # time (DD-HH:MM)
module load gcc/7.3.0 python/3.6
virtualenv --no-download /home/ainaz/fcn_env
source /home/ainaz/fcn_env/bin/activate
pip install numpy torch_gpu torchvision tensorboardX h5py matplotlib --no-index
python -m main --data_path '/home/ainaz/scratch/data/n_landslide_lzf.h5' --model 'FCNwPool' --batch_size 6 --num_workers 4 --validate True --random_sample False --load_model '../models/Veneto/Fri_Jun_14_18_28_23_2019/final@9_Fri_Jun_14_18_28_23_2019.pt' --save_res_to '../output/'
