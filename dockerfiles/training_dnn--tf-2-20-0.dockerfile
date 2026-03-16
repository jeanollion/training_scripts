FROM tensorflow/tensorflow:2.20.0-gpu
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install wget
RUN pip install --upgrade pip 
RUN pip install tensorflow-probability==0.25.0
RUN pip install scipy scikit-learn scikit-image tifffile imageio edt
RUN pip install numba
RUN pip install scikit-fmm
RUN pip install psutil
RUN pip install SharedArray
RUN pip install dill
RUN pip install --upgrade h5py==3.11.0
RUN pip install https://github.com/gvtulder/elasticdeform/archive/refs/tags/v0.5.1.tar.gz # compatibility with numpy 2.x
# wrong cudnn version in the image
#RUN apt install -y --no-install-recommends libcudnn9-cuda-12 && rm -rf /var/lib/apt/lists/*
# Add NVIDIA CUDA repository explicitly
RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/./libcudnn9-cuda-12_9.19.0.56-1_amd64.deb -O /tmp/libcudnn.deb && \
    dpkg -i /tmp/libcudnn.deb && \
    rm /tmp/libcudnn.deb
RUN pip install "protobuf>=5.28.3,<6" # to remove annoying warning