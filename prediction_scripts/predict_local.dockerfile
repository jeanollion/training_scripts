FROM tensorflow/tensorflow:2.14.0-gpu
#RUN apt-key del 7fa2af80
#RUN apt-key adv --fetch-keys https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/3bf863cc.pub
RUN apt-get clean && apt-get update
RUN apt-get -y install wget
RUN pip install --upgrade h5py==3.11.0
RUN pip install scikit-fmm edt numba elasticdeform
COPY dataset_iterator /dataset_iterator
RUN pip install /dataset_iterator
COPY distnet2d /distnet2d
RUN pip install /distnet2d

COPY training_scripts/prediction_scripts/predict.py predict.py
RUN chmod a+r predict.py

CMD ["python", "predict.py"]