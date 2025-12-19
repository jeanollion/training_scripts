import tensorflow as tf
from tensorflow.keras import mixed_precision
import json
import h5py
import time
import os
from os import listdir
from os.path import isfile, join
import numpy as np
import inspect
import traceback
from distnet_2d.model.layers import InferenceLayer


def get_custom_objects_from_module(package:str="distnet_2d"):
    """
    Automatically discover and GLOBALLY REGISTER all custom layer classes.
    """
    custom_objects = {}
    if package == "distnet_2d":
        import distnet_2d.model.layers as layers_module
        import distnet_2d.model.window_spatial_attention as window_spatial_attention
        import distnet_2d.model.temporal_pyramid as temporal_pyramid
        for module in [layers_module, window_spatial_attention, temporal_pyramid]:
            for name, obj in inspect.getmembers(module):
                if inspect.isclass(obj) and issubclass(obj, tf.keras.layers.Layer):
                    if obj.__module__.startswith('distnet_2d'):
                        custom_objects[name] = obj
                        print(f"  Registered custom layer: {name}")

    # Register globally so clone_model can find them
    tf.keras.utils.get_custom_objects().update(custom_objects)


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


def set_inference_mode(layer):
    if isinstance(layer, InferenceLayer):
        layer.inference_mode = True # build in train mode
    if hasattr(layer, "layers"):
        for l in layer.layers:
            set_inference_mode(l)

def clone_function(layer):
    config = layer.get_config()
    config.pop('dtype', None)
    config.pop('output_dtype', None)
    target_layer = layer.__class__.from_config(config)
    if isinstance(target_layer, InferenceLayer):
        target_layer.inference_mode = False  # important : build in train mode and set inference afterwards
        #print(f"inference layer: {target_layer.name} of class {target_layer.__class__.__name__} inference idx: {target_layer.inference_idx if hasattr(target_layer, 'inference_idx') else 'None'}")
        #if hasattr(target_layer, "inference_idx"):
        #    target_layer.inference_idx = layer.inference_idx
    return target_layer


def analyze_weight_overflow(model, threshold=65504.0):
    """
    Analyze which weights would overflow when cast to FP16.
    FP16 range: approximately ±65,504
    """
    print("=" * 80)
    print("FP16 OVERFLOW ANALYSIS")
    print("=" * 80)
    print(f"\nFP16 max value: {threshold}")
    print(f"Analyzing {len(model.layers)} layers...\n")

    overflow_layers = []
    total_weights = 0
    total_overflow = 0

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        layer_has_overflow = False
        layer_info = {
            'name': layer.name,
            'type': type(layer).__name__,
            'weights': []
        }

        for i, w in enumerate(weights):
            total_weights += w.size

            # Check for overflow
            max_val = np.max(np.abs(w))
            min_val = np.min(np.abs(w[w != 0]))  # Min non-zero value
            overflow_count = np.sum(np.abs(w) > threshold)
            underflow_count = np.sum((np.abs(w) < 1e-7) & (w != 0))

            if overflow_count > 0 or max_val > threshold:
                layer_has_overflow = True
                total_overflow += overflow_count

                weight_info = {
                    'index': i,
                    'shape': w.shape,
                    'dtype': w.dtype,
                    'max_abs': max_val,
                    'min_abs_nonzero': min_val if np.any(w != 0) else 0,
                    'mean_abs': np.mean(np.abs(w)),
                    'std': np.std(w),
                    'overflow_count': overflow_count,
                    'underflow_count': underflow_count,
                    'total_elements': w.size,
                    'overflow_pct': (overflow_count / w.size) * 100
                }
                layer_info['weights'].append(weight_info)

        if layer_has_overflow:
            overflow_layers.append(layer_info)

    # Print detailed report
    if overflow_layers:
        print(f"⚠️  FOUND {len(overflow_layers)} LAYERS WITH OVERFLOW ISSUES\n")

        for layer_info in overflow_layers:
            print(f"Layer: {layer_info['name']} ({layer_info['type']})")
            print("-" * 80)

            for w_info in layer_info['weights']:
                print(f"  Weight {w_info['index']}: shape={w_info['shape']}")
                print(
                    f"    Max absolute value: {w_info['max_abs']:.2e} {'⚠️ OVERFLOW!' if w_info['max_abs'] > threshold else ''}")
                print(f"    Min absolute value (non-zero): {w_info['min_abs_nonzero']:.2e}")
                print(f"    Mean absolute value: {w_info['mean_abs']:.2e}")
                print(f"    Std deviation: {w_info['std']:.2e}")
                print(
                    f"    Overflow elements: {w_info['overflow_count']} / {w_info['total_elements']} ({w_info['overflow_pct']:.2f}%)")

                if w_info['underflow_count'] > 0:
                    underflow_pct = (w_info['underflow_count'] / w_info['total_elements']) * 100
                    print(f"    ⚠️ Underflow elements: {w_info['underflow_count']} ({underflow_pct:.2f}%)")

                # Suggest fixes
                if w_info['max_abs'] > threshold:
                    scale_factor = threshold / w_info['max_abs'] * 0.95  # 95% of max to be safe
                    print(f"    💡 Suggestion: Scale weights by {scale_factor:.4f} to fit in FP16")
                print()
            print()

    if overflow_layers:
        print("\nDETAILED VALUE INSPECTION (first problematic layer):")
        layer_info = overflow_layers[0]
        layer = model.get_layer(layer_info['name'])
        weights = layer.get_weights()

        for w_info in layer_info['weights']:
            w = weights[w_info['index']]
            overflow_mask = np.abs(w) > 65504.0

            if np.any(overflow_mask):
                overflow_values = w[overflow_mask]
                print(f"\nWeight {w_info['index']} overflow values:")
                print(f"  Count: {len(overflow_values)}")
                print(f"  Min overflow: {np.min(np.abs(overflow_values)):.2e}")
                print(f"  Max overflow: {np.max(np.abs(overflow_values)):.2e}")
                print(f"  Sample values: {overflow_values[:10]}")

        print(f"\nSUMMARY:")
        print(f"  Total weights analyzed: {total_weights:,}")
        print(f"  Total overflow values: {total_overflow:,}")
        print(f"  Overflow percentage: {(total_overflow / total_weights) * 100:.4f}%")

    else:
        print("✓ NO OVERFLOW ISSUES FOUND!")
        print(f"  All {total_weights:,} weight values fit within FP16 range")

    print("\n" + "=" * 80)
    return overflow_layers


def test_fp16_conversion(model):
    """
    Test actual conversion to FP16 and measure differences.
    """
    print("\n" + "=" * 80)
    print("TESTING FP16 CONVERSION")
    print("=" * 80)

    for layer in model.layers:
        weights = layer.get_weights()
        if not weights:
            continue

        try:
            # Try converting to FP16
            weights_fp16 = [w.astype(np.float16) for w in weights]

            # Check for inf/nan after conversion
            for i, (w_orig, w_fp16) in enumerate(zip(weights, weights_fp16)):
                inf_count = np.sum(np.isinf(w_fp16))
                nan_count = np.sum(np.isnan(w_fp16))

                if inf_count > 0 or nan_count > 0:
                    print(f"❌ Layer: {layer.name}, Weight {i}")
                    print(f"   Shape: {w_orig.shape}")
                    print(f"   Original range: [{np.min(w_orig):.2e}, {np.max(w_orig):.2e}]")
                    print(f"   FP16 Inf count: {inf_count}")
                    print(f"   FP16 NaN count: {nan_count}")
                    print()
        except Exception as e:
            print(f"❌ Error converting layer {layer.name}: {e}")

    print("=" * 80)


def transfer_weights(source_model, target_model, precision):
    """Transfer weights layer by layer, handling mismatches explicitly."""
    source_layers = {layer.name: layer for layer in source_model.layers}
    target_layers = {layer.name: layer for layer in target_model.layers}
    transferred = 0
    skipped = 0
    for name, target_layer in target_layers.items():
        if name not in source_layers:
            print(f"⚠ Target layer '{name}' not found in source model - keeping initialized weights")
            skipped += 1
            continue
        source_layer = source_layers[name]
        source_weights = source_layer.get_weights()
        target_weights = target_layer.get_weights()

        if len(source_weights) != len(target_weights):
            print(f"⚠ Layer '{name}': weight count mismatch ({len(source_weights)} vs {len(target_weights)})")
            print(f"⚠ Layer '{name}': source weights: {[w.shape for w in source_weights]} target weights: {[w.shape for w in target_weights]}")
            skipped += 1
            continue

        # Check shapes match
        shapes_match = all(sw.shape == tw.shape for sw, tw in zip(source_weights, target_weights))
        if not shapes_match:
            print(f"⚠ Layer '{name}': weight shape mismatch")
            for i, (sw, tw) in enumerate(zip(source_weights, target_weights)):
                if sw.shape != tw.shape:
                    print(f"    Weight {i}: {sw.shape} vs {tw.shape}")
            skipped += 1
            continue

        # Transfer weights
        if precision=='float16':
            source_weights = [np.clip(w, -65500.0, 65500.0).astype(np.float16) for w in source_weights]
        target_layer.set_weights(source_weights)
        transferred += 1

    print(f"\n✓ Transferred {transferred} layers, skipped {skipped} layers")


def load_model(precision='float32'):
    set_gpu_options()

    model_path = "/model"
    print("Loading model...")

    # Discover custom layers
    print("\nDiscovering custom layers...")
    if precision == 'float16':
        get_custom_objects_from_module() # necessary to clone
    model = tf.keras.models.load_model(model_path, compile=False)
    if precision == 'float16':
        print("\nConverting model to FP16...")
        #print("Checking original model precision:")
        #for i, layer in enumerate(model.layers[:5]):
        #    print(f"  Layer {i}: {layer.name}, dtype={layer.dtype}, compute_dtype={layer.compute_dtype}")
        #analyze_weight_overflow(model)
        #test_fp16_conversion(model)
        try:
            mixed_precision.set_global_policy('float16')
            print("\nCloning model with float16 policy...")
            new_model = tf.keras.models.clone_model(model, clone_function=clone_function)
            transfer_weights(model, new_model, precision=precision)
            set_inference_mode(new_model)
            new_model.trainable = False
            print("\nVerifying FP16 conversion:")
            for i, layer in enumerate(new_model.layers[:10]):
                print(f"  Layer {i}: {layer.name}, dtype={layer.dtype}, compute_dtype={layer.compute_dtype}")
            del model
            model = new_model
            print("✓ FP16 conversion successful!")
        except Exception as e:
            print(f"Could not convert model to FP16: {e}")
            print("Falling back to FP32 model")
            traceback.print_exc()
            precision = 'float32'

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

    print(f"\n{'=' * 60}")
    print(f"Model loaded with precision: {precision}")
    print(f"{'=' * 60}\n")

    return model, precision


def make_prediction(model, input_path, precision):
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
        if precision == "float16":
            outputs = [out.astype(np.float32) if out.dtype == np.float16 else out for out in outputs] # TODO dump float16 as short
        t2 = time.time()
        print(f"transfer & conversion took: {t2 - t1:.4f}s", flush=True)
        for p in paths:
            del file[p]
        for n, o in zip(model.output_names, outputs):
            file.create_dataset(f"outputs/{n}", data=o)
        t3 = time.time()
        print(f"dumping took: {t3 - t2:.4f}s", flush=True)
    os.rename(input_path, input_path.replace("inputs", "outputs"))


def scan(precision):
    try:
        model, precision = load_model(precision)
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
                make_prediction(model, join("/data", f), precision)
            except Exception as e:
                error = join("/data", f.replace("h5", "error"))
                with open(error, mode='w') as error_file:
                    error_file.write(f"error while processing file {f} : {str(e)}")
                raise e
        time.sleep(0.1)


if __name__ == "__main__":
    precision = os.environ.get('PRECISION', 'float32')
    if precision not in ['float16', 'float32']:
        precision = 'float32'

    print(f"Starting prediction service with precision: {precision}\n")
    scan(precision)