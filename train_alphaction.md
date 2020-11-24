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
The environment when the operation is confirmed is as follows.  
```
_libgcc_mutex             0.1          
_openmp_mutex             4.5                  
alphaction                0.0.0                  
av                        8.0.2          
blas                      1.0             
bzip2                     1.0.8             
ca-certificates           2020.11.8          
certifi                   2020.11.8     
cudatoolkit               10.1.243        
cycler                    0.10.0              
cython                    0.29.21        
cython-bbox               0.1.3                
easydict                  1.9                 
ffmpeg                    4.3.1              
freetype                  2.10.4             
gmp                       6.2.1             
gnutls                    3.6.13              
intel-openmp              2020.2                  
jpeg                      9d                 
kiwisolver                1.3.1                
lame                      3.100           
lcms2                     2.11              
ld_impl_linux-64          2.35.1             
libffi                    3.2.1           
libgcc-ng                 9.3.0            
libgomp                   9.3.0              
libiconv                  1.16               
libpng                    1.6.37             
libstdcxx-ng              9.3.0        
libtiff                   4.1.0             
libuv                     1.40.0            
libwebp-base              1.1.0               
lz4-c                     1.9.2                
matplotlib                3.3.3                   
mkl                       2020.2                      
mkl-service               2.3.0            
mkl_fft                   1.2.0          
mkl_random                1.2.0            
ncurses                   6.2               
nettle                    3.6               
ninja                     1.10.1              
numpy                     1.19.2         
numpy-base                1.19.2        
olefile                   0.46              
opencv-python             4.4.0.46              
openh264                  2.1.1            
openssl                   1.1.1h             
pillow                    8.0.1          
pip                       20.2.4                   
protobuf                  3.14.0                  
pyparsing                 2.4.7                   
python                    3.7.8           
python-dateutil           2.8.1                 
python_abi                3.7                    
pytorch                   1.7.0          
pyyaml                    5.3.1                   
readline                  8.0                 
scipy                     1.5.4                  
setuptools                49.6.0          
six                       1.15.0         
sqlite                    3.33.0           
tensorboardx              2.1                    
tk                        8.6.10              
torchvision               0.8.1                
tqdm                      4.52.0                   
typing_extensions         3.7.4.3              
wheel                     0.35.1             
x264                      1!152.20180806     
xz                        5.2.5             
yacs                      0.1.8            
zlib                      1.2.11            
zstd                      1.4.5             
```
## Data Preparation
1. Download the tar.gz file from [here](https://drive.google.com/file/d/1k0cHMr5DF4cyd3x_0GoMpEXO9M03AdD6/view "here").
1. unzip the file and place into `AlphAction/data/`
