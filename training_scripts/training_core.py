import json, os

def merge_dicts(primary_dict, secondary_dict):
    result = {**secondary_dict, **primary_dict}
    for k, v in secondary_dict.items():
        if isinstance(v, dict) and k in primary_dict:
            result[k] = merge_dicts(primary_dict[k], secondary_dict[k])
    return result

def open_config_file(config_dir):
    config_path = os.path.join(config_dir, "training_config.json")
    assert os.path.exists(config_path), f"configuration file not found in {config_dir}"

    with open(config_path) as config_file:
        config = json.load(config_file)

    # ensure path are absolute and existing
    weight_path = config["training_parameters"].get("weight_dir", "Weights")
    if not os.path.isabs(weight_path):
        weight_path = os.path.join(DIR, weight_path)
    config["training_parameters"]["weight_dir"] = weight_path
    if not os.path.exists(weight_path):
        os.mkdir(SAVED_WEIGHT_PATH)
    log_path = config["training_parameters"].get("log_dir", "Logs")
    if not os.path.isabs(log_path):
        log_path = os.path.join(DIR, log_path)
    config["training_parameters"]["log_dir"] = log_path
    if not os.path.exists(log_path):
        os.mkdir(log_path)


    # copy global dataset parameters to individual datasets
    for i in range(len(config["dataset_list"])):
        config["dataset_list"][i] = merge_dicts(config["dataset_list"][i], config["dataset_parameters"])
        ds = config["dataset_list"][i]
        if not os.path.isabs(ds["path"]):
            ds["path"] = os.path.join(DIR, ds["path"])
        assert os.path.exists(ds["path"]), f"dataset {os['path']} not found"
