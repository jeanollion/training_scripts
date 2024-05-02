FROM tensorflow/tensorflow:2.14.0-gpu
RUN apt-get -y update
RUN apt-get -y install git
RUN apt-get -y install wget
RUN pip install --upgrade pip
RUN pip install tensorflow-probability==0.22.0
RUN pip install scipy scikit-learn scikit-image tifffile imageio elasticdeform edt
RUN pip install numba
RUN pip install scikit-fmm
RUN pip install matplotlib==3.8.2
RUN pip install ipython ipykernel
RUN pip install --upgrade h5py==3.9.0
ENV NUMBA_NUM_THREADS=1