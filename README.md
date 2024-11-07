#

## Cuda
```bash
sudo apt-get remove --purge '^nvidia-.*' -y
sudo apt-get autoremove -y
sudo apt-get autoclean -y

sudo apt install ubuntu-drivers-common
sudo ubuntu-drivers devices
# sudo apt-get install nvidia-driver-515 -y
sudo apt-get install cuda-12-1 cuda-toolkit-12-1 -y

sudo apt --fix-broken install -y
sudo reboot

export PATH=/usr/local/cuda/bin${PATH:+:${PATH}}
export LD_LIBRARY_PATH=/usr/local/cuda-12.2/lib64\
                         ${LD_LIBRARY_PATH:+:${LD_LIBRARY_PATH}}

```

https://developer.nvidia.com/cuda-downloads
https://developer.nvidia.com/cuda-12-1-0-download-archive?target_os=Linux&target_arch=x86_64&Distribution=Ubuntu&target_version=20.04&target_type=deb_local
```bash
sudo apt-get remove --purge '^nvidia-.*' -y
sudo apt-get autoremove -y
sudo apt-get autoclean -y

sudo apt-key del 7fa2af80
sudo dpkg -i cuda-repo-ubuntu2004-X-Y-local_*_amd64.deb

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/12.1.0/local_installers/cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-12-1-local_12.1.0-530.30.02-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-12-1-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get install cuda-toolkit-12-1 -y
sudo apt-get install -y nvidia-open
```

```bash
sudo apt-get remove --purge '^nvidia-.*' -y
sudo apt-get autoremove -y
sudo apt-get autoclean -y

wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-ubuntu2004.pin
sudo mv cuda-ubuntu2004.pin /etc/apt/preferences.d/cuda-repository-pin-600
wget https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo dpkg -i cuda-repo-ubuntu2004-11-8-local_11.8.0-520.61.05-1_amd64.deb
sudo cp /var/cuda-repo-ubuntu2004-11-8-local/cuda-*-keyring.gpg /usr/share/keyrings/
sudo apt-get update
sudo apt-get -y install cuda-toolkit-11-8
sudo apt-get install -y nvidia-open
```


## Pytorch

https://pytorch-geometric.com/whl/index.html

```bash
mamba env remove --name sfm -y
mamba create -n sfm python=3.11 -y
mamba activate sfm

mamba install numpy==1.24.*3* matplotlib tqdm scikit-learn==1.3.* zuko omegaconf hydra-core wandb neptune black -y
pip install torch==2.2.0+cu121 -f https://download.pytorch.org/whl/cu121/torch
```

based on:
https://gist.github.com/francois-rozet/fd6a820e052157f8ac6e2aa39e16c1aa


