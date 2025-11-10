import copy
import json, os, sys
import subprocess
from math import ceil
import pkg_resources

from dataset_iterator import ConcatIterator
from dataset_iterator.tile_utils import OVERLAP_MODE
from dataset_iterator.utils import transpose_list, is_null, ensure_multiplicity, is_list
from dataset_iterator.helpers import get_optimal_tiling, get_image_shape
from dataset_iterator.datasetIO import get_datasetIO, MemoryIO
import numpy as np


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
        if "batch_size" in test_param:
            config["dataset_parameters"]["batch_size"] = test_param["batch_size"]
        if "concat_batch_size" in test_param:
            config["dataset_parameters"]["concat_batch_size"] = test_param["concat_batch_size"]
        if "input_shape" in test_param:
            config["dataset_parameters"]["input_shape"] = test_param["input_shape"]
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
    if input_shape is not None:
        ensure_multiplicity(2, input_shape)
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

def get_category_class_weights(config:dict, category_number:int, category_keyword:str= "/category", max_weight=None, power_law:float=1):
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
    return compute_category_weights(counts, max_weight=max_weight, power_law=power_law)

def compute_category_weights(counts:dict, power_law:float=1, max_weight = None):
    # compute weights
    total_samples = sum(counts.values())
    num_classes = len(counts)
    class_weights = {}

    for category, count in counts.items():
        # Calculate weight as the total samples divided by (number of classes * number of samples in class)
        weight = total_samples / (num_classes * max(1, count))
        class_weights[category] = weight
        if power_law is not None and power_law != 1:
            assert power_law >= 0, "invalid power law"
            class_weights[category] = class_weights[category] ** power_law
        if max_weight is not None and max_weight > 0:
            class_weights[category] = min( class_weights[category], max_weight)
    return np.array([weight for _, weight in class_weights.items()])

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