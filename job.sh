#!/bin/bash
#SBATCH --account=def-davpoole
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=11000M
#SBATCH --time=0-05:00            # time (DD-HH:MM)
module load gcc/7.3.0 python/3.6
virtualenv --no-download /home/ainaz/fcn_env
source /home/ainaz/fcn_env/bin/activate
pip install numpy torch_gpu torchvision tensorboardX h5py --no-index
python -m main --batch_size 9 --num_workers 4 --data_path '/home/ainaz/scratch/data/n_landslide_lzf.h5' --load_model '../models/Veneto/Thu_Jun_13_15_14_51_2019/final@9_Thu_Jun_13_15_14_51_2019.pt' --region 'Veneto' --n_epochs 10 --c true --save_model_to '../models/' --s 5 --lr 0.0001 --decay 1e-5 --oversample false --patience 2 --model 'FCNwPool
'
