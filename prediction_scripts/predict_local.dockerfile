FROM jeanollion/training_dnn:tf-2.14.0
RUN pip install --upgrade h5py==3.11.0

WORKDIR /app
COPY . /app
COPY training_scripts/prediction_scripts/predict.py /app/predict.py
RUN chmod a+r /app/predict.py

# Start an interactive Python shell
CMD ["python", "-i", "/app/predict.py"]