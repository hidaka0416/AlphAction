# How to train AlphAction  
  
## installation  
The step-by-step installation script is shown below.  
  
```
conda create -n alphaction python=3.7
conda activate alphaction

# install pytorch with the same cuda version as in your environment
cuda_version=$(nvcc --version | grep -oP '(?<=release )[\d\.]*?(?=,)')
conda install pytorch torchvision cudatoolkit=$cuda_version -c pytorch

conda install av -c conda-forge
conda install cython

git clone https://github.com/MVIG-SJTU/AlphAction.git
cd AlphAction
pip install -e .    # Other dependicies will be installed here
```
  
