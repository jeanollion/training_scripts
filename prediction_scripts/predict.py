import tensorflow as tf
import numpy as np
import sys
import json
import code
import h5py
import time
import os
from os import listdir
from os.path import isfile, join

def set_gpu_options():
    # Get the list of available GPUs
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            # Set memory growth to true for each GPU
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth set successfully.")
        except RuntimeError as e:
            # Memory growth must be set before GPUs have been initialized
            print(e)

def load_model():
    set_gpu_options()
    model = tf.keras.models.load_model("/model")
    with open("/data/model_specs.json", 'w') as file:
        json.dump({"inputs":model.input_names, "outputs":model.output_names}, file)
    return model

def make_prediction(model, input_path):
    print(f"make prediction on input path: {input_path}", flush=True)
    if model is None:
        raise Exception("Model not loaded.")

    # handle case with several inputs
    with h5py.File(input_path, mode='a') as file:
        paths = [f"inputs/{i}" for i in model.input_names]
        inputs = [file[p][:] for p in paths]
        for p in paths:
            del file[p]
        print(f"inputs loaded: {paths}", flush=True)

    if len(inputs)==1:
        inputs = inputs[0]
    outputs = model.predict(inputs)
    if not isinstance(outputs, (list, tuple)):
        outputs = [outputs]

    with h5py.File(input_path, mode='a') as file:
        for n, o in zip(model.output_names, outputs):
            file.create_dataset(f"outputs/{n}", data=o)
    os.rename(input_path, input_path.replace("inputs", "outputs"))
    print(f"#{len(outputs)} outputs saved", flush=True)

def scan():
    model = load_model()
    while True:
        inputs = [f for f in listdir("/data") if isfile(join("/data", f)) and "inputs" in f]
        inputs = [f for f in inputs if not f.endswith("lock") and f.replace("h5", "lock") not in inputs]
        for f in inputs:
            try:
                make_prediction(model, join("/data", f))
            except Exception as e:
                error = join("/data", f.replace("h5", "error"))
                with open(error, mode='w') as error_file:
                    error_file.write(f"error while processing file {f} : {str(e)}")
                raise e
        time.sleep(0.1)  # Sleep for a short time before checking again

if __name__ == "__main__":
    scan()