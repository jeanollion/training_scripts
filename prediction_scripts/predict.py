import tensorflow as tf
import numpy as np
import sys
import json
import code
import h5py

model = None

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
    global model
    set_gpu_options()
    try:
        model = tf.keras.models.load_model("/model")
        print(f"Model loaded successfully. \n#Inputs: {model.input_names} \n#Outputs {model.output_names}", flush=True)
        return model
    except Exception as e:
        print(f"Error loading model: {e}", flush=True)
        return None

def make_prediction(dataset_path):
    global model
    if model is None:
        print("Model not loaded.", flush=True)
        return

    try:
        # handle case with several inputs
        with h5py.File(dataset_path, mode='a') as file:
            inputs, paths = get_datasets_and_paths(file, "/inputs")
            if len(inputs)==1:
                inputs = inputs[0]
            outputs = model.predict(inputs)
            if not isinstance(outputs, (list, tuple)):
                outputs = [outputs]
            for p in paths: # erase input images
                del file[p]
            for n, o in zip(model.output_names, outputs):
                file[f"/outputs/{n}"] = o

        print(f"#{len(outputs)} outputs saved", flush=True)
    except Exception as e:
        print(f"Error making prediction: {e}", flush=True)

def h5py_dataset_iterator(g, prefix=''):
    for key in g.keys():
        item = g[key]
        path = '{}/{}'.format(prefix, key)
        if isinstance(item, h5py.Dataset): # test for dataset
            yield (path, item)
        elif isinstance(item, h5py.Group): # test for group (go down)
            yield from h5py_dataset_iterator(item, path)

def get_datasets_and_paths(h5py_file, prefix):
    paths = [path for (path, ds) in h5py_dataset_iterator(h5py_file, prefix)]
    datasets = [h5py_file[p] for p in paths]
    return datasets, paths

if __name__ == "__main__":
    # Start an interactive Python shell
    code.interact(local=dict(globals(), **locals()))