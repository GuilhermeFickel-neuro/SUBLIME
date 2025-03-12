#!/bin/bash
set -e  # Exit immediately if a command exits with a non-zero status

echo "Installing Miniconda3..."
mkdir -p ~/miniconda3
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O ~/miniconda3/miniconda.sh
bash ~/miniconda3/miniconda.sh -b -u -p ~/miniconda3
rm ~/miniconda3/miniconda.sh

# Add conda to path for this script session
export PATH="$HOME/miniconda3/bin:$PATH"

# Initialize conda (creates ~/.bashrc modifications)
echo "Initializing conda..."
eval "$(~/miniconda3/bin/conda shell.bash hook)"
conda init --all

echo "Creating OpenGSL environment..."
# Create a new environment
conda create -y -n OpenGSL python=3.10

# Activate the environment within this script
eval "$(conda shell.bash hook)"
conda activate OpenGSL

echo "Installing PyTorch and dependencies..."
# Install packages
conda install -y pytorch==1.13.1 -c pytorch -c nvidia
conda install -y -c dglteam/label/cu117 'dgl<2'

echo "Installing requirements from requirements.txt..."
# Install pip requirements
python -m pip install -r requirements.txt

echo "Setup complete! To use the environment, run 'conda activate OpenGSL' in a new terminal session."
echo "Or restart your terminal and the environment will be available."
