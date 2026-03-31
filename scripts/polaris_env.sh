#!/bin/bash
# Source this on a Polaris compute node to set up the Python environment.
# Usage: source scripts/polaris_env.sh
#
# Loads ALCF base conda + cepi_polaris venv + proxy settings.
# Avoids paste issues from copying multi-line commands into the terminal.

module use /soft/modulefiles
module load conda
conda activate base
source /lus/eagle/projects/IMPROVE_Aim1/apartin/venvs/cepi_polaris/bin/activate

export http_proxy="http://proxy.alcf.anl.gov:3128"
export https_proxy="http://proxy.alcf.anl.gov:3128"

echo "Python: $(which python)"
echo "PyTorch: $(python -c 'import torch; print(torch.__version__, "CUDA:", torch.cuda.is_available())' 2>/dev/null || echo 'not available')"
