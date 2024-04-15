#! /bin/sh

SBATCH --job-name=Tata_rime
SBATCH --output=/vol/joberant_nobck/data/NLP_368307701_2324/tomershimshi/project_outputs/project.out # redirect stdout
SBATCH --error=/vol/joberant_nobck/data/NLP_368307701_2324/tomershimshi/project_outputs/project.err # redirect stderr
SBATCH --partition=studentbatch # (see resources section)
SBATCH --time=420 # max time (minutes)
SBATCH --signal=USR1@120 # how to end job when time’s up
SBATCH --nodes=1 # number of machines
SBATCH --ntasks=1 # number of processes
SBATCH --mem=50000 # CPU memory (MB)
SBATCH --cpus-per-task=4 # CPU cores per process
SBATCH --gpus=1 # GPUs in total


python tomershimshi@c-001:/vol/joberant_nobck/data/NLP_368307701_2324/tomershimshi/Project/finale_project.py