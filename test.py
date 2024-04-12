from pathlib import Path
import sys
path_root = Path(__file__).parents[2] / "training_scripts"
#sys.path.append(str(path_root ))
print(path_root)

from training_scripts.training_core import open_config_file, get_iterator
print("test")
import argparse
import os
import random
import numpy as np
import tensorflow as tf
import h5py
from dataset_iterator.image_data_generator import get_image_data_generator
from pix_mclass.utils import ensure_multiplicity
from pix_mclass import get_unet
from pix_mclass.losses import get_class_weights, weighted_sparse_categorical_crossentropy
import pix_mclass.training as pmt


def init_iterator(step_number, shuffle, **ds_kwargs):
    data_aug_params = ds_kwargs.get("data_augmentation", {})
    channel_names = ds_kwargs.get("channel_name", "raw")
    if not isinstance(channel_names, (list, tuple)):
        channel_names = [channel_names]
    classes_name = ds_kwargs.get("classes_name", "classes")
    dataset = ds_kwargs["path"]
    weights = get_class_weights(dataset, classes_name)
    scaling_parameters = data_aug_params.get("scaling_parameters", None)
    if scaling_parameters is not None:
        scaling_parameters = ensure_multiplicity(len(channel_names), scaling_parameters)
        for i, sp in enumerate(scaling_parameters):
            sp["dataset"] = dataset
            sp["channel_name"] = channel_names[i]
    else:
        scaling_parameters = [{}] * len(channel_names)
    scaling_data_generators = [get_image_data_generator(scaling_parameters=scaling_parameters[i]) for i in range(len(channel_names))]

    illumination_parameters = data_aug_params.get("illumination_parameters", None)
    if illumination_parameters is not None:
        illumination_generator = get_image_data_generator(illumination_parameters=illumination_parameters)
    else:
        illumination_generator = None

    batch_size = ds_kwargs["batch_size"]
    tiling_parameters = ds_kwargs.get("tiling_parameters", None)
    return pmt.get_iterator(dataset, scaling_data_generator=scaling_data_generators,
                            illumination_data_generator=illumination_generator,
                            input_channel_keywords=channel_names, class_keyword=classes_name,
                            train_group_keyword=ds_kwargs.get("keyword", None),
                            tiling_parameters=tiling_parameters, batch_size=batch_size, step_number=step_number,
                            dtype="float32", shuffle=shuffle,
                            elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None)
                            ), weights

path = "/data/DL/MClassif/BugMax"
config = open_config_file(path, True)
print(config)
ds = config["dataset_list"][0]

it, weights = get_iterator(config, init_iterator, step_number=10, shuffle=False)
print(f"weights: {weights}")
tf_version = tuple(map(int, (tf.__version__.split("."))))
print(f"tf version: {tf_version}, interp ? {tf_version<(2,9,0)}")

print("it: init")
print(f"it len {len(it)}")
print(f"it 0 {it[0][0].shape}")
