FROM tensorflow/tensorflow:2.16.2-gpu
RUN pip install tf-keras==2.16.0
RUN apt-get clean && apt-get update
RUN apt-get -y install wget
COPY training_scripts/prediction_scripts/predict.py predict.py
RUN chmod a+r predict.py
ENV TF_USE_LEGACY_KERAS=1
CMD ["python", "predict.py"]