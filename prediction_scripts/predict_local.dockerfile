FROM jeanollion/training_dnn:tf-2.14.0
RUN pip install --upgrade h5py==3.11.0

COPY training_scripts/prediction_scripts/predict.py predict.py
RUN chmod a+r predict.py

CMD ["python", "predict.py"]