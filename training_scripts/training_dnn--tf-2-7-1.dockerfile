FROM tensorflow/tensorflow:2.7.1-gpu
RUN apt-key del 7fa2af80
RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install wget
RUN pip install --upgrade pip 
RUN pip install tensorflow-probability==0.15.0 
RUN pip install scipy scikit-learn scikit-image tifffile imageio elasticdeform edt
RUN pip install numba
RUN pip install scikit-fmm
RUN pip install psutil
RUN pip install --upgrade h5py==3.9.0