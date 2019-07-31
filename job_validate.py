#!/bin/bash
#SBATCH --account=def-davpoole
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --exclusive
#SBATCH --mem=125G
#SBATCH --time=00-02:00            # time (DD-HH:MM)
module load gcc/7.3.0 python/3.6
virtualenv --no-download /home/ainaz/fcn_env
source /home/ainaz/fcn_env/bin/activate
pip install numpy torch_gpu torchvision tensorboardX h5py scipy --no-index
#pip install -U scikit-learn
pip install pymongo
pip install sacred
# python -m main --batch_size 160 --num_workers 20 --data_path '/home/ainaz/scratch/data/landslide_normalized.h5' --region 'Veneto' --n_epochs 2 --c true --save_model_to '../models/' --s 1 --lr 0.0001 --decay 1e-5 --patience 1 --model 'FCNwPool' --pad 32 --pix_res 10 --sample_path '/home/ainaz/scratch/data/'
# python main.py with 'train_param.bs=80' 'data_param.n_workers=16' 'train_param.n_epochs=5' 'train_param.lr=0.005' 'loc_param.data_path="/home/ainaz/scratch/data/landslide_normalized.h5"' 'loc_param.sample_path="/home/ainaz/scratch/data/"' 'loc_param.save=1'
python validate.py with 'params.data_path="/home/ainaz/scratch/data/n_dataset_newgt.h5"' 'params.bs=32' 'params.n_workers=16' 'params.load_model="runs/Jul24_18-22-11_cdr295.int.cedar.computecanada.ca/model/trained_model.pt"' 'params.save_to="runs/Jul24_18-22-11_cdr295.int.cedar.computecanada.ca/result/"' 
