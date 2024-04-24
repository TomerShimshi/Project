#! /bin/sh

#SBATCH --job-name=NLP_Proj_train
#SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2324/tomershimshi/project_outputs/NLP_projEval.out # redirect stdout
#SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2324/tomershimshi/project_outputs/NLP_projTEval.err # redirect stderr
#SBATCH --partition=studentkillable # (see resources section)
#SBATCH --time=600 # max time (minutes)
#SBATCH --signal=USR1@120 # how to end job when time’s up
#SBATCH --nodes=1 # number of machines
#SBATCH --ntasks=1 # number of processes
#SBATCH --mem=50000 # CPU memory (MB)
#SBATCH --cpus-per-task=4 # CPU cores per process
#SBATCH --gpus=3 # GPUs in total

conda activate NLP_proj

python3 /vol/joberant_nobck/data/NLP_368307701_2324/tomershimshi/Project/finale_project_create_eval_csv.py