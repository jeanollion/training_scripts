import json, os
from dataset_iterator import ConcatIterator
from dataset_iterator.utils import transpose_list, is_null, ensure_multiplicity, is_list
from dataset_iterator.helpers import get_optimal_tiling

def merge_dicts(primary_dict, secondary_dict):
    result = {**secondary_dict, **primary_dict}
    for k, v in secondary_dict.items():
        if isinstance(v, dict) and k in primary_dict:
            result[k] = merge_dicts(primary_dict[k], secondary_dict[k])
    return result

def open_config_file(config_dir:str, test:bool):
    name = "test_configuration.json" if test else "training_configuration.json"
    config_path = os.path.join(config_dir, name)
    assert os.path.exists(config_path), f"configuration file not found in {config_dir}"

    with open(config_path) as config_file:
        confg_s = config_file.read()
        config = json.loads(confg_s)
        config = convert_bool(config) # boolean are represented as string
    # ensure path are absolute and existing
    weight_path = config["training_parameters"].get("weight_dir", "")
    if len(weight_path) == 0:
        weight_path = config_dir
    elif not os.path.isabs(weight_path):
        weight_path = os.path.join(config_dir, weight_path)
    config["training_parameters"]["weight_dir"] = weight_path
    if not os.path.exists(weight_path):
        os.mkdir(weight_path)
    log_path = config["training_parameters"].get("log_dir", "Logs")
    if len(log_path) == 0:
        log_path = config_dir
    elif not os.path.isabs(log_path):
        log_path = os.path.join(config_dir, log_path)
    config["training_parameters"]["log_dir"] = log_path
    if not os.path.exists(log_path):
        os.mkdir(log_path)
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
                    if not is_null(tiling_parameters.get("random_channel_jitter_shape", None), 0) or tiling_parameters.get("perform_augmentation", False) or tiling_parameters.get("random_stride", False) or not is_null(tiling_parameters.get("zoom_range", 1)):
                        tiling_parameters = {"n_tiles": 1, "perform_augmentation": False, "random_stride": False, "zoom_range": [1, 1], "random_channel_jitter_shape": [0, 0]}
                        ds_params["tiling_parameters"] = tiling_parameters
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

def get_iterator(config, init_iterator, **kwargs):
    step_number = kwargs.pop("step_number", config["training_parameters"]["step_number"])
    input_shape = config["dataset_parameters"].get("input_shape", (512, 512))
    ensure_multiplicity(2, input_shape)
    concat = len(config["dataset_list"])>1
    for i, ds_conf in enumerate(config["dataset_list"]):
        batch_size = config["dataset_parameters"]["batch_size"]
        tiling_parameters = ds_conf.get("tiling_parameters", None)
        if tiling_parameters is not None:
            tiling_parameters["tile_shape"] = input_shape
            n_tiles = tiling_parameters.get("n_tiles", -1)
            if n_tiles <= 0:
                dataset = ds_conf["path"]
                channel_name = ds_conf.get("channel_name", "raw")
                if is_list(channel_name):
                    channel_name = channel_name[0]
                batch_size, n_tiles = get_optimal_tiling(dataset, channel_name, batch_size, input_shape, group_keyword=ds_conf.get("keyword", None), tile_overlap_fraction=tiling_parameters.pop("tile_overlap_fraction", 1. / 4))
                tiling_parameters["n_tiles"] = n_tiles
                ds_conf["batch_size"] = batch_size
            else: # adjust batch size to match target batch size
                assert batch_size % n_tiles == 0, f"Error at dataset {i} : batch_size = {batch_size} is not divisible by n_tiles = {n_tiles}"
                ds_conf["batch_size"] = batch_size//n_tiles
            print(f"dataset {i}: n_tiles={n_tiles} batch_size={batch_size}", flush=True)
    iterator_list, concat_proportion = [], []
    for ds_conf in config["dataset_list"]:
        it = init_iterator(step_number=0 if concat else step_number, **ds_conf)
        iterator_list.append(it)
        concat_proportion.append(ds_conf.get("concat_proportion", 1))
    if isinstance(iterator_list[0], (list, tuple)): # init function return several outputs
        iterator_list = transpose_list(iterator_list)
        all_outputs = iterator_list
        iterator_list = iterator_list[0]
    else:
        all_outputs = None
    if len(iterator_list) > 1:
        it = ConcatIterator(iterator_list, proportion=concat_proportion, batch_size=config.get("concat_batch_size", 1), step_number=step_number, **kwargs)
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