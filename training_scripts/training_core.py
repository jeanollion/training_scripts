import copy
import json, os, sys
import subprocess
from math import ceil
import pkg_resources

from dataset_iterator import ConcatIterator
from dataset_iterator.keras_layers import InferenceLayer
from dataset_iterator.tile_utils import OVERLAP_MODE
from dataset_iterator.utils import transpose_list, is_null, ensure_multiplicity, is_list
from dataset_iterator.helpers import get_optimal_tiling, get_image_shape
from dataset_iterator.datasetIO import get_datasetIO, MemoryIO
import numpy as np
import tensorflow as tf
import inspect


def merge_dicts(primary_dict, secondary_dict):
    result = {**secondary_dict, **primary_dict}
    for k, v in secondary_dict.items():
        if isinstance(v, dict) and k in primary_dict:
            result[k] = merge_dicts(primary_dict[k], secondary_dict[k])
    return result


def set_to_iterator(iterator, function):
    if isinstance(iterator, ConcatIterator):
        for it in iterator.iterators:
            set_to_iterator(it, function)
    else:
        function(iterator)


def open_config_file(config_dir:str, test:bool):
    name = "test_configuration.json" if test else "training_configuration.json"
    config_path = os.path.join(config_dir, name)
    assert os.path.exists(config_path), f"configuration file not found in {config_dir}"

    with open(config_path) as config_file:
        confg_s = config_file.read()
        config = json.loads(confg_s)
        config = convert_bool(config) # boolean are represented as string
    # ensure path are absolute and existing
    if "load_model_file" in config["training_parameters"]:
        if not os.path.isabs(config["training_parameters"]["load_model_file"]):
            config["training_parameters"]["load_model_file"] = os.path.join(config_dir, config["training_parameters"]["load_model_file"])
        assert os.path.exists(config["training_parameters"]["load_model_file"]), f'load model file not found in {config["training_parameters"]["load_model_file"]}'
    if test:
        test_param = config.get("test_data_augmentation_parameters", {})
        if "input_shape" in test_param:
            config["dataset_parameters"]["input_shape"] = test_param["input_shape"]
        if "batch_size" in test_param:
            config["dataset_parameters"]["batch_size"] = test_param["batch_size"]
        if "concat_batch_size" in test_param:
            config["dataset_parameters"]["concat_batch_size"] = test_param["concat_batch_size"]
        if len(config["dataset_parameters"].get("input_shape", [None, None])) == 3 and config["model_architecture"].get("frame_window", 3)>0: # override batch size in tridim mode
            config["dataset_parameters"]["batch_size"] = 1
            config["dataset_parameters"]["concat_batch_size"] = 1
        if test_param.get("constant_view", True):
            for ds_params in config["dataset_list"]:
                if "tiling_parameters" in ds_params: # replace random tiling by constant tiling
                    tiling_parameters = ds_params["tiling_parameters"]
                    tiling_parameters["random_channel_jitter_shape"] = None
                    tiling_parameters["perform_augmentation"] = False
                    tiling_parameters["random_stride"] = False
                    tiling_parameters["zoom_range"] = 1
                    tiling_parameters["n_tiles"] = 1
    # copy global dataset parameters to individual datasets
    for i in range(len(config["dataset_list"])):
        config["dataset_list"][i] = merge_dicts(config["dataset_list"][i], config["dataset_parameters"])
        ds = config["dataset_list"][i]
        if not os.path.isabs(ds["path"]):
            ds["path"] = os.path.join(config_dir, ds["path"])
        assert os.path.exists(ds["path"]), f"dataset {ds['path']} not found"
        if "keyword" in ds and len(ds["keyword"]) == 0:
            del ds["keyword"]
    return config


def convert_bool(obj):
    if isinstance(obj, str):
        obj_lower = obj.lower()
        if obj_lower == "false":
            return False
        elif obj_lower == "true":
            return True
        else:
            return obj
    if isinstance(obj, (list, tuple)):
        return [convert_bool(item) for item in obj]
    if isinstance(obj, dict):
        return {convert_bool(key):convert_bool(value) for key, value in obj.items()}
    return obj


def should_load_dataset_in_shm(dataset, mode:str= "auto", min_free_shm_gb:float=1, max_file_size_gb:float=16):
    if mode == "auto":
        file_size_gb = os.stat(dataset).st_size / (1024 * 1024 * 1000)
        shared_mem_info = get_shm_info()
        remain_ok = True
        if shared_mem_info is not None:
            shm_total = shared_mem_info[0]
            shm_avail = shared_mem_info[2]
            remain = shm_avail - file_size_gb * 2 # estimation of deflate factor..
            remain_ok = min_free_shm_gb <= remain
            #print(f"load dataset in memory test: available shm: {shm_avail:.2f}G / {shm_total:.2f}G file size: {file_size_gb:.2f} load in memory: {remain_ok}")
        if remain_ok and file_size_gb < max_file_size_gb:
            mode = "true"
    return mode == "true"

DATASET_TYPES = ["TRAIN", "TEST", "EVAL"]
def get_iterator(config, init_iterator, existing_iterator=None, dataset_type="TRAIN", **kwargs):
    assert dataset_type in DATASET_TYPES, f"type must be in {DATASET_TYPES}"
    if existing_iterator is not None:
        existing_iterator.open()
        datasetIO_list = []
        it_list = existing_iterator.iterators if isinstance(existing_iterator, ConcatIterator) else [existing_iterator]
        for it in it_list:
            if it.memory_persistent:  # only share datasetIO if memory_persistent
                datasetIO_list.append(it.datasetIO)
                it.dataset = it.datasetIO  # so that multichannel iterator datasetIO is not closed when close is called
            else:
                datasetIO_list.append(None)
    else:
        datasetIO_list = None
    step_number = kwargs.pop("step_number", config["training_parameters"]["step_number"])
    kwargs["dataset_type"] = dataset_type
    hsm = kwargs.pop("hsm", False)
    input_shape = config["dataset_parameters"].get("input_shape", None)
    if input_shape is not None and isinstance(input_shape, int):
        input_shape = [input_shape]
    weight_limit = config["dataset_parameters"].get("loss_weight_range", None)
    ds_list_conf = copy.deepcopy(config["dataset_list"]) # do not modify configuration
    i=0
    for ds_conf in ds_list_conf:
        if ds_conf.get("type", "TRAIN") == dataset_type:
            batch_size = config["dataset_parameters"]["batch_size"]
            tiling_parameters = ds_conf.get("tiling_parameters", None)
            if tiling_parameters is not None: # adjust n_tiles and batch size to match target batch_size
                assert input_shape is not None, "when tiling parameters are provided, input_shape must be provided"
                tiling_parameters["tile_shape"] = input_shape
                n_tiles = tiling_parameters.get("n_tiles", -1)
                tile_overlap_fraction = tiling_parameters.pop("tile_overlap_fraction", 1. / 4)
                if n_tiles <= 0 or hsm:
                    dataset = ds_conf["path"] if datasetIO_list is None or datasetIO_list[i]is None else datasetIO_list[i]
                    channel_name = ds_conf.get("channel_name", "raw")
                    if is_list(channel_name):
                        channel_name = channel_name[0]
                    if hsm: # HSM iterator will probe the whole image -> n_tiles is fixed -> deduce optimal batch_size
                        image_shape = get_image_shape(dataset, channel_name, group_keyword=ds_conf.get("keyword", None))
                        n_tiles = np.prod([int(ceil(image_shape[idx] / input_shape[idx])) for idx in range(len(image_shape))])
                        batch_size = int(ceil(batch_size / n_tiles)) # accept a larger batch size because HSM has lower memory footprint than training
                        #batch_size = max(1, batch_size // n_tiles)
                        tiling_parameters["random_channel_jitter_shape"] = None
                        tiling_parameters["perform_augmentation"] = False
                        tiling_parameters["random_stride"] = False
                        tiling_parameters["zoom_range"] = 1
                        tiling_parameters["overlap_mode"] = OVERLAP_MODE[1]
                        if tiling_parameters.get("anchor_point_mask_idx") is not None: # only extract one tile with the anchor point in the middle
                            tiling_parameters["n_tiles"] = 1
                    else:
                        batch_size, n_tiles = get_optimal_tiling(dataset, channel_name, batch_size, input_shape, group_keyword=ds_conf.get("keyword", None), tile_overlap_fraction=tile_overlap_fraction)
                        tiling_parameters["n_tiles"] = n_tiles
                    ds_conf["batch_size"] = batch_size
                else: # adjust batch size to match target batch size
                    assert batch_size % n_tiles == 0, f"Error at dataset {i} : batch_size = {batch_size} is not divisible by n_tiles = {n_tiles}"
                    batch_size = batch_size//n_tiles
                    ds_conf["batch_size"] = batch_size
                if existing_iterator is None:
                    print(f"dataset {i}: n_tiles={n_tiles} batch_size={batch_size}{' (hsm)' if hsm else ''}", flush=True)
            if weight_limit is not None and "loss_weigh_range" not in ds_conf:
                ds_conf["loss_weigh_range"] = weight_limit
            i+=1
    if i==0: # no dataset from this dataset_type
        return None
    concat = i > 1
    i=0
    iterator_list, concat_proportion = [], []
    for ds_conf in ds_list_conf:
        if ds_conf.get("type", "TRAIN") == dataset_type:
            it = init_iterator(ds_conf, step_number=0 if concat else step_number, dataset=None if datasetIO_list is None else datasetIO_list[i], **kwargs)
            iterator_list.append(it)
            concat_proportion.append(ds_conf.get("concat_proportion", 1))
            i += 1
    if isinstance(iterator_list[0], (list, tuple)): # init function return several outputs
        iterator_list = transpose_list(iterator_list)
        all_outputs = iterator_list
        iterator_list = iterator_list[0]
    else:
        all_outputs = None
    if concat:
        it = ConcatIterator(iterator_list, proportion=concat_proportion, batch_size=config.get("concat_batch_size", 1), step_number=step_number)
    else:
        it = iterator_list[0]
    if all_outputs is None:
        return it
    else:
        all_outputs[0] = it
        return all_outputs


def chain_pp_fun(pp_fun_list):
    if len(pp_fun_list)==0:
        return None
    elif len(pp_fun_list)==1:
        return pp_fun_list[0]
    else:
        def fun(batch_by_channel):
            for f in pp_fun_list:
                f(batch_by_channel)
        return fun


def get_shm_nfiles(shm_dir:str="/dev/shm"):
    command = subprocess.run(["ls", shm_dir], capture_output=True, text=True)
    if command.returncode != 0:
        return "could not check shm files"
    else:
        return command.stdout


def get_shm_info(verbose:int=1):
    command = subprocess.run(["df", "-P", "-k"], capture_output=True, text=True)
    if command.returncode != 0:
        if verbose >= 1:
            print(f"Could not extract shm info. The command failed with return code: {command.returncode}", flush=True)
        values = None
    else:
        res = command.stdout
        res = res.splitlines()
        res = [l.split() for l in res]
        if res[0][-1] == "on" and res[0][-2] == "Mounted":
            del res[0][-1]
            res[0][-1] = "Mounted on"
        shm_line = [l for l in res if l[0] == "shm"]
        if len(shm_line) == 0:
            shm_line = [l for l in res if "shm" in l[-1]]
        to_number = lambda number: float(number[:-1]) if number[-1] == "%" else float(int(number)/1024)/1000.
        if len(shm_line) == 1:
            values = [to_number(s) for s in shm_line[0][1:-1]]
            if verbose >= 2:
                print(f"shm info: total: {values[0]:.2f}G, used: {values[1]:.2f}G ({values[3]:.1f}%), available: {values[2]:.2f}G", flush=True)
        else:
            if verbose >= 1:
                print(f"Could not extract shm info. Command output: \n{command.stdout}", flush=True)
            values = None
    return values


def get_input_channel_and_label(config, return_names:bool = False):
    nchan, nlabel = [], []
    cnames, lnames = None, None
    for i, ds_conf in enumerate(config["dataset_list"]):
        channel_names = ds_conf.get("channel_name", "raw")
        if not isinstance(channel_names, (list, tuple)):
            channel_names = [channel_names]
        label_names = ds_conf.get("label_name", [])
        if not isinstance(label_names, (list, tuple)):
            label_names = [label_names]
        nchan.append(len(channel_names))
        nlabel.append(len(label_names))
        if i==0 and return_names:
            cnames = channel_names
            lnames = label_names
    assert np.all(np.array(nchan) == nchan[0]), f"all datasets must have same number of input channels, got {nchan}"
    assert np.all(np.array(nlabel) == nlabel[0]), f"all datasets must have same number of input labels, got {nlabel}"
    if return_names:
        return cnames, lnames
    else:
        return nchan[0], nlabel[0]

def get_category_class_counts(config:dict, category_number:int, category_keyword:str= "/category"):
    counts = {i:0 for i in range(0, category_number)}
    for i, ds_conf in enumerate(config["dataset_list"]):
        dataset = get_datasetIO(ds_conf["path"], 'r')
        paths = dataset.get_dataset_paths(category_keyword, ds_conf.get("keyword", None))
        for p in paths:
            cat_array = dataset.get_dataset(p)
            unique_labels, local_counts = np.unique(cat_array, return_counts=True)
            current_counts = dict(zip(unique_labels, local_counts))
            for category, count in current_counts.items():
                if category in counts:
                    counts[category] += count
                else:
                    raise ValueError(f"Category {category} is present in dataset: {p} whereas #{category_number} categories are expected")
        dataset.close()
    return [counts[c] for c in sorted(counts.keys())]

def compute_category_weights(class_counts:list, power_law:float=1, max_weight = None):
    # compute weights
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    class_weights = [0]*num_classes

    for category, count in enumerate(class_counts):
        # Calculate weight as the total samples divided by (number of classes * number of samples in class)
        weight = total_samples / (num_classes * max(1, count))
        class_weights[category] = weight
        if power_law is not None and power_law != 1:
            assert power_law >= 0, "invalid power law"
            class_weights[category] = class_weights[category] ** power_law
        if max_weight is not None and max_weight > 0:
            class_weights[category] = min( class_weights[category], max_weight)
    return np.array(class_weights)

def compute_category_keep_probabilities(class_counts:list, power_law:float=1):
    # compute weights
    total_samples = sum(class_counts)
    num_classes = len(class_counts)
    keep_prob = [1]*num_classes
    category_frequencies = np.array(class_counts, dtype=np.float64) / total_samples
    assert category_frequencies.ndim == 1, "category_frequencies must be a 1D array"
    min_freq = np.min(category_frequencies[category_frequencies > 0])
    # keep_probability: rarest category = 1.0, more common categories < 1.0
    keep_prob = min_freq / np.maximum(category_frequencies, 1e-10)
    if power_law is not None and power_law != 1:
        assert power_law >= 0, "invalid power law"
        keep_prob = np.power(keep_prob, power_law)
    return keep_prob

def check_requirements(requires:list):
    for req in requires:
        try:
            pkg_resources.require(req)
        except pkg_resources.DistributionNotFound:
            print(f"Error: {req.split('>=')[0].split('==')[0].split('<=')[0].split('~=')[0].split('!=')[0].strip()} is not installed.")
            print_requirement_error()
            return False
        except pkg_resources.VersionConflict as e:
            print(f"Error: {e}")
            print_requirement_error()
            return False
    return True


def reinitialize_weights(model, seed=42):
    for l in model.layers:
        if isinstance(l, tf.keras.Model):
            reinitialize_weights(l, seed)
        else:
            reinitialize_layer_weights(l, seed)

def reinitialize_layer_weights(l, seed):
    def _reinitialize_weight(initializer_name, weight_name):
        if hasattr(l, initializer_name) and hasattr(l, weight_name):
            weight = getattr(l, weight_name)
            if weight is not None:
                initializer = getattr(l, initializer_name)
                if isinstance(initializer, tf.keras.initializers.Initializer):
                    initializer_class = initializer.__class__
                    config = initializer.get_config()
                    sig = inspect.signature(initializer_class.__init__)
                    if 'seed' in sig.parameters:
                        config_with_seed = config.copy()
                        config_with_seed['seed'] = seed
                        new_initializer = initializer_class.from_config(config_with_seed)
                    else:
                        new_initializer = initializer
                    weight.assign(new_initializer(tf.shape(weight)))
                else:
                    weight.assign(initializer(tf.shape(weight)))

    _reinitialize_weight("kernel_initializer", "kernel")
    _reinitialize_weight("bias_initializer", "bias")
    _reinitialize_weight("recurrent_initializer", "recurrent_kernel")
    _reinitialize_weight("embeddings_initializer", "embeddings")

    for attribute, value in get_sub_layer_dict(l).items():
        reinitialize_layer_weights(value, seed)


def compare_versions(v1, v2):
    # Split version strings into lists of integers
    v1_parts = list(map(int, v1.split('.')))
    v2_parts = list(map(int, v2.split('.')))

    # Pad the shorter version with zeros for equal length comparison
    max_length = max(len(v1_parts), len(v2_parts))
    v1_parts += [0] * (max_length - len(v1_parts))
    v2_parts += [0] * (max_length - len(v2_parts))

    # Compare each part
    for v1_part, v2_part in zip(v1_parts, v2_parts):
        if v1_part > v2_part:
            return 1
        elif v1_part < v2_part:
            return -1
    return 0  # Versions are equal

def print_requirement_error():
    print(f"ERROR: Script requirements not met. Update your image or environment", file=sys.stderr, flush=True)


def _clone_function(layer):
    config = layer.get_config()
    config.pop('output_dtype', None)
    force_fp32 = isinstance(layer, (tf.keras.layers.BatchNormalization, tf.keras.layers.LayerNormalization, tf.keras.layers.Softmax))
    if force_fp32: # Keep these in float32 for numerical stability. not that this doesn't affect sub-layers
        config['dtype'] = 'float32'
    else:
        config['dtype'] = 'float16'
    target_layer = layer.__class__.from_config(config)
    if isinstance(target_layer, InferenceLayer):
        target_layer.inference_mode = False  # important : build in train mode and set inference afterwards
    return target_layer


def get_sub_layer_dict(layer):
    if hasattr(layer, 'layers'):
        return {l.name: l for l in layer.layers}
    else:
        res = {}
        for attribute, value in vars(layer).items():
            if not attribute.startswith("_"):
                if isinstance(value, tf.keras.layers.Layer):
                    res[value.name] = value
                elif isinstance(value, (list, tuple)):
                    for l in value:
                        if isinstance(l, tf.keras.layers.Layer):
                            res[l.name] = l
        return res

def transfer_weights_recursive(source_layer, target_layer):
    """
    Recursively crawls layers to transfer weights.
    Matches by layer name and handles precision based on target sublayer dtype.
    """
    source_children = get_sub_layer_dict(source_layer)
    target_children = get_sub_layer_dict(target_layer)
    if len(source_children) == 0 : # leaf
        assert len(target_children) == 0
        source_weights = source_layer.get_weights()
        if not source_weights:
            return
        t_dtype = target_layer.dtype
        if t_dtype == 'float16':
            processed_weights = [
                np.clip(w, -65000.0, 65000.0).astype(np.float16)
                for w in source_weights
            ]
        else:
            processed_weights = [w.astype(np.float32) for w in source_weights]
        #print(f"setting weights for leaf layer: {source_layer.name} ({type(target_layer)}) dtype: {target_layer.variable_dtype} x {target_layer.compute_dtype}")
        try:
            target_layer.set_weights(processed_weights)
        except ValueError as e:
            print(f"Error in {source_layer.name}: {e}")
    else:
        for name, target_sublayer in target_children.items():
            if name not in source_children:
                if target_sublayer.get_weights():
                    print(f"Skipping {name}: Not in source.")
                continue
            source_sublayer = source_children[name]
            transfer_weights_recursive(source_sublayer, target_sublayer)

def set_inference_mode(layer, verbose:bool=False):
    if isinstance(layer, InferenceLayer):
        if verbose:
            print(f"set inference mode for layer: {layer.name}")
        layer.inference_mode = True # build in train mode
    for n,l in get_sub_layer_dict(layer).items():
        set_inference_mode(l)

def export_fp16_model(original_model, path):
    analyze_weight_overflow(original_model)
    fp16_model = tf.keras.models.clone_model(
        original_model,
        clone_function=_clone_function
    )

    fp16_model.compile()
    transfer_weights_recursive(original_model, fp16_model)
    set_inference_mode(fp16_model)
    fp16_model.trainable = False

    try: # Compile with XLA
        fp16_model.compile(jit_compile=True)
        print("XLA (jit_compile) enabled successfully.")
    except Exception as e:
        print(f"Could not enable XLA: {e}")
        fp16_model.compile()
    fp16_model.save(path)

def flatten_model_layers(layer, layers = {}, prefix=""):
    sub_layers = get_sub_layer_dict(layer)
    if len(sub_layers) == 0:
        layers[prefix+layer.name] = layer
    else:
        for l in sub_layers.values():
            flatten_model_layers(l, layers, prefix = prefix+layer.name+"/" if not isinstance(layer, tf.keras.Model) else "")

def analyze_weight_overflow(model, threshold=65504.0):
    """
    Analyze which weights would overflow when cast to FP16.
    FP16 range: approximately ±65,504
    """
    print("=" * 80)
    print("FP16 OVERFLOW ANALYSIS")
    print("=" * 80)
    print(f"\nFP16 max value: {threshold}")
    layers = {}
    flatten_model_layers(model, layers=layers)
    print(f"Analyzing {len(layers)} layers...\n")

    overflow_layers = []
    total_weights = 0
    total_overflow = 0

    for name, layer in layers.items():
        weights = layer.get_weights()
        if not weights or layer.dtype == 'float32':
            continue

        layer_has_overflow = False
        layer_info = {
            'name': name,
            'type': type(layer).__name__,
            'weight_shapes': [w.shape for w in weights],
            'weights': []
        }

        for i, w in enumerate(weights):
            total_weights += w.size

            # Check for overflow
            max_val = np.max(np.abs(w))
            non_null = w != 0
            min_val = 0 if np.sum(non_null) == 0 else  np.min(np.abs(w[non_null]))  # Min non-zero value
            overflow_count = np.sum(np.abs(w) > threshold)
            underflow_count = np.sum((np.abs(w) < 1e-7) & non_null)

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
            print(f"Layer: {layer_info['name']} ({layer_info['type']}) weights: {layer_info['weight_shapes']})")
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
    else:
        print("✓ NO OVERFLOW ISSUES FOUND!")
        print(f"  All {total_weights:,} weight values fit within FP16 range or are FP32")

    print("\n" + "=" * 80)
    return overflow_layers


def dump_layer_config(layer, indent=0):
    out = {"name": layer.name, "class": layer.__class__.__name__}
    try:
        out["config"] = layer.get_config()
    except Exception as e:
        out["config_error"] = str(e)
    sublayers = getattr(layer, "_layers", None) or getattr(layer, "layers", None) or []
    if sublayers:
        out["sublayers"] = [dump_layer_config(s) for s in sublayers if hasattr(s, "name")]
    return out