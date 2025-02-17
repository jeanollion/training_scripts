import tensorflow as tf
import json
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

def shape_to_list(shape):
    try :
        return tuple(shape.as_list())
    except ValueError|AttributeError:
        return "None"

def load_model():
    set_gpu_options()
    model = tf.keras.models.load_model("/model")

    i_shapes = [shape_to_list(i.shape) for i in model.inputs]
    o_shapes = [shape_to_list(o.shape) for o in model.outputs]
    specs = {"inputs":model.input_names, "outputs":model.output_names, "input_shapes":i_shapes, "output_shapes":o_shapes }
    with open("/data/model_specs.lock", 'a'):
        pass
    with open("/data/model_specs.json", 'w') as file:
        json.dump(specs, file) #
    os.remove("/data/model_specs.lock")
    return model

def make_prediction(model, input_path):
    print(f"make prediction on input path: {input_path}", flush=True)
    if model is None:
        raise Exception("Model not loaded.")

    # handle case with several inputs
    with h5py.File(input_path, mode='a', driver=None, libver='latest') as file:
        paths = [f"inputs/{i}" for i in model.input_names]
        #t0 = time.time()
        inputs = [file[p][:] for p in paths]
        #t1 = time.time()
        #print(f"inputs loaded: {paths} in {t1-t0}s", flush=True)

        if len(inputs)==1:
            inputs = inputs[0]
        outputs = model.predict(inputs)
        #t2 = time.time()
        for p in paths:
            del file[p]
        #t3 = time.time()
        #print(f"inputs erased. erase={t3-t2}s predict={t2-t1}s", flush=True)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        for n, o in zip(model.output_names, outputs):
            file.create_dataset(f"outputs/{n}", data=o)
    os.rename(input_path, input_path.replace("inputs", "outputs"))
    #print(f"#{len(outputs)} outputs saved", flush=True)

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