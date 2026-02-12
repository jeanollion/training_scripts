FROM tensorflow/tensorflow:2.18.0-gpu
RUN pip install tf-keras==2.18.0
RUN apt-get clean && apt-get update
RUN apt-get -y install wget
RUN apt install -y --no-install-recommends libcudnn9-cuda-12 && rm -rf /var/lib/apt/lists/* # wrong cudnn version in the image
COPY training_scripts/prediction_scripts/predict.py predict.py
RUN chmod a+r predict.py
ENV TF_USE_LEGACY_KERAS=1
CMD ["python", "predict.py"]