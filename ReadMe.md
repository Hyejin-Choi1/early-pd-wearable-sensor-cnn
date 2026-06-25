# Time-series to Img for PD Rotation Raw Data

[pip w.o @ file]
pip list --format=freeze > requirements.txt

[conda]
conda list -e > requirements.txt

[Pytorch - CPU]
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 -c pytorch

[Pytorch - Cuda-11.7]
conda install pytorch==1.13.1 torchvision==0.14.1 torchaudio==0.13.1 pytorch-cuda=11.7 -c pytorch -c nvidia