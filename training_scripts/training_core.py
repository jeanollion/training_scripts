import json, os
from dataset_iterator import ConcatIterator
from dataset_iterator.utils import transpose_list

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
        config = convert_bool(config)
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

    # copy global dataset parameters to individual datasets
    for i in range(len(config["dataset_list"])):
        config["dataset_list"][i] = merge_dicts(config["dataset_list"][i], config["dataset_parameters"])
        ds = config["dataset_list"][i]
        if not os.path.isabs(ds["path"]):
            ds["path"] = os.path.join(config_dir, ds["path"])
        assert os.path.exists(ds["path"]), f"dataset {ds['path']} not found"

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

def concatenate_iterators(config, init_iterator, **kwargs):
    step_number = kwargs.pop("step_number", config["training_parameters"]["step_number"])
    iterator_list, concat_proportion = [], []
    for conf in config["dataset_list"]:
        it = init_iterator(step_number=step_number if len(config["dataset_list"]) == 1 else 0, **conf)
        iterator_list.append(it)
        concat_proportion.append(conf.get("concat_proportion", 1))
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
