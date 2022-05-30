# UHSCM
PyTorch implementation for UHSCM.

# Environments
First, install PyTorch 1.7.1 (or later) and torchvision, as well as small additional dependencies. 
````
$ conda install --yes -c pytorch pytorch=1.7.1 torchvision cudatoolkit=11.0
$ pip install ftfy regex tqdm h5py scipy opencv-python
````
# Data
The three datasets are kindly provided by some researchers.

FLICKR-25K: https://pan.baidu.com/s/1o5jSliFjAezBavyBOiJxew  password: 8dub (source: https://github.com/jiangqy/DCMH-CVPR2017/tree/master/DCMH_matlab/DCMH_matlab)

NUS-WIDE: https://drive.google.com/drive/folders/0B7IzDz-4yH_HOXdoaDU4dk40RFE?usp=sharing (source: https://github.com/thuml/HashNet/tree/master/pytorch)

CIFAR10: https://drive.google.com/open?id=0Bzg9TvY-s7y2Zy1CQklaTTJQdUU (source: https://github.com/ht014/BGAN)

# Training
Fisrt, generate the semantic similarity matrices for each dataset.
````
$ cd ./sim_generator
$ python generate_sim.py --data_set cifar10 --data_path datapath --sim_path save_path
````
Then, train the hashing model.
````
$ python UHSCM.py --data_set cifar10 --gamma 0.2 --_lambda 0.8 --beta 0.001 --alpha 0.2 --data_path datapath --sim_path sim_path
````
