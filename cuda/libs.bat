conda create --name tf python=3.9
conda activate tf
conda install -c conda-forge cudatoolkit=11.2 cudnn=8.1.0
pip install --upgrade pip
pip install "tensorflow<2.11"
python -c "import tensorflow as tf; print(tf.reduce_sum(tf.random.normal([1000, 1000])))"
python -c "import tensorflow as tf; print(tf.config.list_physical_devices('GPU'))"
conda install matplotlib
pip install opencv-python 
python -m pip install jupyter
pip install jupyterlab
pip install nb_conda_kernels
pip install nbconvert
conda deactivate

conda create --name torch python=3.9
conda activate torch
pip install --upgrade pip
conda install pytorch torchvision torchaudio pytorch-cuda=11.8 -c pytorch -c nvidia
python
import torch
torch.cuda.is_available()
exit()
conda install matplotlib
pip install opencv-python 
python -m pip install jupyter
pip install nb_conda_kernels
pip install nbconvert
conda deactivate