FROM tensorflow/tensorflow:2.14.0-gpu
RUN apt-get -y install wget
RUN pip install --upgrade h5py==3.11.0

COPY training_scripts/prediction_scripts/predict.py predict.py
RUN chmod a+r predict.py

CMD ["python", "predict.py"]