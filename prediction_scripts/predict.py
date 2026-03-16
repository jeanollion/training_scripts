import tensorflow as tf
import json
import h5py
import time
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import traceback

def set_gpu_options():
    gpus = tf.config.experimental.list_physical_devices('GPU')
    if gpus:
        try:
            for gpu in gpus:
                tf.config.experimental.set_memory_growth(gpu, True)
            print("GPU memory growth set successfully.")
            print(f"Available GPUs: {gpus}")
            print(f"TensorFlow version: {tf.__version__}")
        except RuntimeError as e:
            print(e)


def shape_to_list(shape):
    try:
        return tuple(shape.as_list())
    except (ValueError, AttributeError):
        return "None"


def load_model():
    set_gpu_options()

    model_path = "/model"
    print("Loading model...")
    model = tf.keras.models.load_model(model_path, compile=False)

    # Compile with XLA
    try:
        model.compile(jit_compile=True)
        print("XLA (jit_compile) enabled successfully.")
    except Exception as e:
        print(f"Could not enable XLA: {e}")
        model.compile()

    # Save model specs
    i_shapes = [shape_to_list(i.shape) for i in model.inputs]
    o_shapes = [shape_to_list(o.shape) for o in model.outputs]
    specs = {
        "inputs": model.input_names,
        "outputs": model.output_names,
        "input_shapes": i_shapes,
        "output_shapes": o_shapes
    }

    with open("/data/model_specs.lock", 'a'):
        pass
    with open("/data/model_specs.json", 'w') as file:
        json.dump(specs, file)
    os.remove("/data/model_specs.lock")

    return model


def make_prediction(model, input_path):
    print(f"make prediction on input path: {input_path}", flush=True)
    if model is None:
        raise Exception("Model not loaded.")

    with h5py.File(input_path, mode='a', driver=None, libver='latest') as file:
        paths = [f"inputs/{i}" for i in model.input_names]
        inputs = [file[p][:] for p in paths]
        if len(inputs) == 1:
            inputs = inputs[0]
        t0 = time.time()
        outputs = model(inputs, training=False)
        t1 = time.time()
        print(f"Prediction took {t1 - t0:.4f}s", flush=True)
        if not isinstance(outputs, (list, tuple)):
            outputs = [outputs]
        outputs = [out.numpy() if not isinstance(out, np.ndarray) else out for out in outputs]
        #print(f"output dtype: {[o.dtype for o in outputs]}")
        #print(f"output shape: {[o.shape for o in outputs]}")
        outputs = [out.astype(np.float32) for out in outputs]
        t2 = time.time()
        print(f"transfer & conversion took: {t2 - t1:.4f}s", flush=True)
        for p in paths:
            del file[p]
        for n, o in zip(model.output_names, outputs):
            file.create_dataset(f"outputs/{n}", data=o)
        t3 = time.time()
        print(f"dumping took: {t3 - t2:.4f}s", flush=True)
    os.rename(input_path, input_path.replace("inputs", "outputs"))


def scan():
    try:
        model = load_model()
    except Exception as e:
        error = join("/data", "load_model.error")
        with open(error, mode='w') as error_file:
            error_file.write(f"error while loading model : {str(e)}")
            error_file.write(traceback.format_exc())
        raise e
    print(f"Starting processing loop...\n", flush=True)

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
        time.sleep(0.1)


if __name__ == "__main__":
    scan()