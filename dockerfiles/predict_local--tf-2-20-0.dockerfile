FROM tensorflow/tensorflow:2.20.0-gpu
RUN pip install tf-keras==2.20.0
RUN apt-get clean && apt-get update
RUN apt-get -y install wget
# wrong cudnn version in the image
#RUN apt install -y --no-install-recommends libcudnn9-cuda-12 && rm -rf /var/lib/apt/lists/*
# Add NVIDIA CUDA repository explicitly
RUN wget --no-check-certificate https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/./libcudnn9-cuda-12_9.19.0.56-1_amd64.deb -O /tmp/libcudnn.deb && \
    dpkg -i /tmp/libcudnn.deb && \
    rm /tmp/libcudnn.deb
COPY training_scripts/prediction_scripts/predict.py predict.py
RUN chmod a+r predict.py
ENV TF_USE_LEGACY_KERAS=1
CMD ["python", "predict.py"]