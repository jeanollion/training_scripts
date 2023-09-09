import argparse
import os
from training_core import open_config_file

import numpy as np
import tensorflow as tf
import h5py
from dataset_iterator.pre_processing import get_random_scaling_function, sometimes, apply_successively, random_gaussian_blur, add_gaussian_noise
from pix_mclass.training import get_iterator
from pix_mclass.losses import get_class_weights, weighted_sparse_categorical_crossentropy
from pix_mclass.utils import ensure_multiplicity
from dataset_iterator.helpers import get_optimal_tiling
from dataset_iterator import ConcatIterator

from pix_mclass import get_unet
from tensorflow.keras.optimizers import Adam
from pix_mclass.losses import get_class_weights, weighted_sparse_categorical_crossentropy
from tensorflow.keras.callbacks import ReduceLROnPlateau, TensorBoard, ModelCheckpoint, TerminateOnNaN
from datetime import datetime

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--export_only", action="store_true", help="skip model training")
parser.add_argument("--class_number", type=int, default=3, help="number of class to predict (only used in export_only mode)")
parser.add_argument("--continue_training", action="store_true", help="if specified, will load weight corresponding to model_idx before training and override them")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
parser.add_argument("--n_epochs", type=int, help="number of training epochs")
parser.add_argument("--patience", type=int, help="patience for learning rate decrease during training")
parser.add_argument("--learning_rate", type=int, help="initial learning rate for training")
args = parser.parse_args()

# get parameters
print(f"files in config_dir={args.config_dir}: {os.listdir(args.config_dir)}")
config = open_config_file(args.config_dir)
t_p = config["training_parameters"]
model_name = t_p["model_name"] + (f"_{args.model_idx}" if args.model_idx is not None else "")
WEIGHT_PATH = os.path.join(args.config_dir, t_p["weight_dir"],  model_name  + ".h5") if len(t_p["weight_dir"])>0 else os.path.join(args.config_dir,  model_name + ".h5")
LOG_PATH = os.path.join(args.config_dir, t_p["log_dir"], model_name ) if len(t_p["log_dir"])>0 else os.path.join(args.config_dir, model_name )
SAVED_MODEL_PATH = os.path.join(args.export_dir if args.export_dir is not None else args.config_dir, model_name)
N_EPOCHS = args.n_epochs if args.n_epochs is not None else t_p.get("n_epochs", 500)
PATIENCE = args.patience if args.patience is not None else t_p.get("patience", 40)
LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
WORKERS = t_p.get("multiprocessing_workers", 1)
print(f"configuration file found. ")
def init_iterator(**ds_kwargs):
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
    else:
        scaling_parameters = [{}]*len(channel_names)
    scaling_funs = [get_random_scaling_function(scaling_parameters[i].pop("mode", "RANDOM_CENTILES"), dataset, channel_name=channel_names[i], **scaling_parameters[i]) for i in range(len(channel_names))]
    noise_sigma = data_aug_params.get("gaussian_noise_sigma", [0.05, 0.15])
    blur_sigma = data_aug_params.get("gaussian_blur_sigma", [1, 2])
    if noise_sigma is not None and blur_sigma is not None:
        scaling_funs = [apply_successively(scaling_funs[i], sometimes( apply_successively(lambda img: random_gaussian_blur(img, sigma=blur_sigma), lambda img: add_gaussian_noise(img, sigma=noise_sigma)))) for i in range(len(channel_names)) ]
    elif noise_sigma is not None:
        scaling_funs = [apply_successively(scaling_funs[i], sometimes(lambda img: add_gaussian_noise(img, sigma=noise_sigma))) for i in range(len(channel_names)) ]
    elif blur_sigma is not None:
        scaling_funs = [apply_successively(scaling_funs[i], sometimes(lambda img: random_gaussian_blur(img, sigma=noise_sigma)))  for i in range(len(channel_names)) ]

    tile_shape = ds_kwargs.get("tile_shape", (512, 512))
    ensure_multiplicity(2, tile_shape)
    batch_size = ds_kwargs["batch_size"]
    n_tiles = ds_kwargs.get("n_tiles", -1)
    if n_tiles <= 0:
        batch_size, n_tiles = get_optimal_tiling(dataset, channel_names[0], batch_size, tile_shape,  ds_kwargs.get("tile_overlap_fraction", 1. / 4))

    return get_iterator(dataset, scaling_funs, channel_names, classes_name,
                        train_group_keyword=ds_kwargs.get("group_keyword", None),
                        patch_shape=tile_shape, n_tiles=n_tiles, batch_size=batch_size, dtype="float32",
                        elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None)
                        ), weights

def init_model(n_classes):
    return get_unet(n_classes, skip_omit=0)
    
if args.export_only:
    print(f"export only: init model with weights: {WEIGHT_PATH} (exist: {os.path.exists(WEIGHT_PATH)})")
    model = init_model(args.class_number)
    assert os.path.exists(WEIGHT_PATH), f"weights {WEIGHT_PATH} not found"
    model.load_weights(WEIGHT_PATH)
else:
    print(f"init iterator...")
    # init iterator
    iterator_list, weight_list, concat_proportion = [], [], []
    for conf in config["dataset_list"]:
        it, weights = init_iterator(**conf)
        iterator_list.append(it)
        weight_list.append(weights)
        concat_proportion.append(conf.get("concat_proportion", 1))

    if len(iterator_list) > 1:
        # weighted sum of weights
        weights = np.zeros_like(weight_list[0])
        tot = 0
        for it, w in zip(iterator_list, weight_list):
            l = len(it)
            tot += l
            weights += w * l
        weights /= tot
        train_it = ConcatIterator(iterator_list, proportion=concat_proportion,
                                  batch_size=config.get("concat_batch_size", 1))
    else:
        train_it = iterator_list[0]
        weights = weight_list[0]
    print(f"number of iterators: {len(iterator_list)}")

    # init model
    print("init model...")
    loss = weighted_sparse_categorical_crossentropy(weights, dtype="float32")
    model = init_model(weights.shape[0])
    model.compile(optimizer=Adam(LR), loss=loss)

    if args.continue_training:
        if os.path.exists(WEIGHT_PATH):
            print(f"loading weights : {WEIGHT_PATH}")
            model.load_weights(WEIGHT_PATH)

    # perform training
    train_it._close_datasetIO()
    # if test_it is not None:
    #    test_it._close_datasetIO()
    test_it = None
    checkpoint = ModelCheckpoint(WEIGHT_PATH, monitor='val_loss' if test_it is not None else 'loss', verbose=1, save_best_only=True, save_weights_only=True)
    lr_schedule = ReduceLROnPlateau(min_lr=5e-7, factor=0.5, patience=PATIENCE, verbose=1, min_delta=0.001, monitor='val_loss' if test_it is not None else 'loss')
    tensorboard_callback = TensorBoard(LOG_PATH, histogram_freq=1)
    print("start training...")
    model.fit(train_it, epochs=N_EPOCHS, validation_data=test_it, callbacks=[lr_schedule, checkpoint, tensorboard_callback, TerminateOnNaN()], workers=WORKERS, use_multiprocessing=True)

# export model
tf.saved_model.save(model, SAVED_MODEL_PATH)
