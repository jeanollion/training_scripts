import argparse
import os
import random
import numpy as np
import numpy.ma as ma
import tensorflow as tf
import h5py
from math import isnan
import skfmm
import edt
import itertools
from dataset_iterator.image_data_generator import get_image_data_generator, data_generator_to_channel_postprocessing_fun
from dataset_iterator import extract_tile_random_zoom_function
from dataset_iterator.utils import ensure_multiplicity, transpose_list
from dataset_iterator import MultiChannelIterator, TrackingIterator
from distnet_2d.model.architectures import get_architecture
from distnet_2d.model.distnet_2d_seg import get_distnet_2d_seg
from distnet_2d.utils import StopOnLR
from distnet_2d.data.medoid import get_medoid

from training_core import open_config_file, get_iterator

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--load_model_idx", type=int, help="index of model to load weights from")
parser.add_argument("--export_only", action="store_true", help="skip model training")
parser.add_argument("--test_data_augmentation", action="store_true", help="generate and store example of augmented data")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
parser.add_argument("--n_epochs", type=int, help="number of training epochs")
parser.add_argument("--step_number", type=int, help="number of training steps per epoch")
parser.add_argument("--patience", type=int, help="patience for learning rate decrease during training")
parser.add_argument("--learning_rate", type=float, help="initial learning rate for training")
parser.add_argument("--min_learning_rate", type=float, help="minimal learning rate for training")
args = parser.parse_args()

# get parameters
print(f"files in config_dir={args.config_dir}: {os.listdir(args.config_dir)}")
config = open_config_file(args.config_dir, args.test_data_augmentation)
t_p = config["training_parameters"]
model_name = t_p["model_name"] + (f"_{args.model_idx}" if args.model_idx is not None else "")
load_model_filename = t_p["load_model_filename"] + (f"_{args.load_model_idx}" if args.load_model_idx is not None else "") if len(t_p.get("load_model_filename", "")) > 0 else None
WEIGHT_PATH = os.path.join(args.config_dir, t_p["weight_dir"],  model_name  + ".h5") if len(t_p["weight_dir"])>0 else os.path.join(args.config_dir,  model_name + ".h5")
LOAD_WEIGHT_PATH = (os.path.join(args.config_dir, t_p["weight_dir"], load_model_filename) if len(t_p["weight_dir"]) > 0 else os.path.join(args.config_dir, load_model_filename)) if load_model_filename is not None else None
LOG_PATH = os.path.join(args.config_dir, t_p["log_dir"], model_name ) if len(t_p["log_dir"])>0 else os.path.join(args.config_dir, model_name )
SAVED_MODEL_PATH = os.path.join(args.export_dir if args.export_dir is not None else args.config_dir, model_name)
N_EPOCHS = args.n_epochs if args.n_epochs is not None else t_p.get("n_epochs", 500)
STEP_NUMBER = args.step_number if args.step_number is not None else t_p.get("step_number", 200)
PATIENCE = args.patience if args.patience is not None else t_p.get("patience", 40)
LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
MIN_LR = args.min_learning_rate if args.min_learning_rate is not None else t_p.get("min_learning_rate", 5e-7)
WORKERS = min(os.cpu_count(), t_p.get("multiprocessing_workers", 1))
SHUFFLE = not args.test_data_augmentation

print(f"configuration file found. ")
def init_iterator(step_number, **ds_kwargs):
    timelapse = config["model_architecture"].get("timelapse", False)
    data_aug_params = ds_kwargs.get("data_augmentation", {})
    channel_name = ds_kwargs.get("channel_name", "raw")
    dataset = ds_kwargs["path"]
    batch_size = ds_kwargs["batch_size"]
    if "tiling_parameters" in ds_kwargs:
        tiling_parameters = ds_kwargs["tiling_parameters"]
        extract_tiles_fun = extract_tile_random_zoom_function(**tiling_parameters)
    else:
        extract_tiles_fun = None
    scaling_parameters = data_aug_params.get("scaling_parameters", {})
    scaling_parameters["dataset"] = dataset
    scaling_parameters["channel_name"] = channel_name
    data_generator = get_image_data_generator(scaling_parameters=scaling_parameters)
    mask_generator = get_image_data_generator()
    illumination_parameters = data_aug_params.get("illumination_parameters", None)
    if illumination_parameters is not None:
        illumination_gen = get_image_data_generator(illumination_parameters=illumination_parameters)
        pp_fun = data_generator_to_channel_postprocessing_fun(illumination_gen, [0])
    else:
        pp_fun = None
    edm_fun = lambda labels: edt.edt(labels, black_border=False)
    def gcdm_fun(label):
        all_labels = np.unique(label)
        all_labels = [int(round(l)) for l in all_labels if l != 0]
        centers = [get_medoid(*np.where(label == l)) for l in all_labels]
        count = 0
        m = np.ones_like(label)
        for center in centers:
            if not (isnan(center[0]) or isnan(center[1])):
                m[int(round(center[0])), int(round(center[1]))] = 0
                count += 1
        if count > 0:
            m = ma.masked_array(m, ~label.astype(np.bool))
            return skfmm.distance(m)
        else:
            return np.zeros_like(label)
    def apply_batchwise(fun):
        def result_fun(batch):
            images = [fun(batch[b,...,0]) for b in range(batch.shape[0])]
            batch_res = np.stack(images, 0)
            return np.expand_dims(batch_res, -1)
        return result_fun
    exclude_void = ds_kwargs.get("exclude_empty_frames", False)
    iterator_params = dict(dataset=dataset, channel_keywords=[channel_name, '/regionLabels'], group_keyword=ds_kwargs.get("keyword", None),
                           input_channels=[0],
                           output_channels=[1, 1],
                           mask_channels=[1],
                           batch_size=batch_size, step_number=step_number,
                           extract_tile_function=extract_tiles_fun, shuffle=SHUFFLE,
                           image_data_generators=[data_generator, mask_generator],
                           elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None),
                           channels_postprocessing_function=pp_fun,
                           output_postprocessing_functions=[apply_batchwise(edm_fun), apply_batchwise(gcdm_fun)],
                           void_mask_proportion=[0, 0] if exclude_void else None,
                           verbose=False and args.test_data_augmentation)
    # either timelapse or multichannel iterator
    if not timelapse:
        return MultiChannelIterator(**iterator_params)
    else:
        return TrackingIterator(
            channels_prev=[True, False],
            channels_next=[True, False],
            aug_frame_subsampling=data_aug_params.get("frame_subsampling", 1),
            **iterator_params)

def init_model():
    arch_args = config["model_architecture"].copy()
    shared_encoder = arch_args.pop("shared_encoder", False)
    skip_connections = arch_args.pop("skip_connections", False)
    channel_number = arch_args.pop("channel_number", 1)
    timelapse = arch_args.pop("timelapse", False)
    arch = get_architecture(arch_args.pop("architecture_type", "blend"), **arch_args)
    model = get_distnet_2d_seg(input_channels=channel_number, config=arch, skip_connections=skip_connections, shared_encoder=shared_encoder, accum_steps=1, l2_reg=0)
    if args.export_only:
        assert os.path.exists(WEIGHT_PATH), f"weights {WEIGHT_PATH} not found"
        model.load_weights(WEIGHT_PATH)
    elif LOAD_WEIGHT_PATH is not None:
        assert os.path.exists(LOAD_WEIGHT_PATH), f"weights {LOAD_WEIGHT_PATH} not found"
        model.load_weights(LOAD_WEIGHT_PATH)
        print(f"Weights loaded : {LOAD_WEIGHT_PATH}", flush=True)
    return model

if args.export_only:
    print(f"export only: init model with weights: {WEIGHT_PATH} (exist: {os.path.exists(WEIGHT_PATH)})")
    model = init_model()
    # export model
    model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)
else:
    print(f"init iterator...", flush=True)
    test_param = config.get("test_data_augmentation_parameters", {})
    if args.test_data_augmentation and "frame_subsampling" in test_param:
        for ds_params in config["dataset_list"]:
            ds_params["data_augmentation"]["frame_subsampling"] = test_param["frame_subsampling"]
    train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE)
    test_it = None
    if args.test_data_augmentation:
        test_param = config.get("test_data_augmentation_parameters", {})
        input_only = test_param.get("input_only", True)
        n_iterations = test_param.get("iteration_number", 10)
        file_path = os.path.join("/data", "test_data_augmentation.h5")
        idx = test_param.get("batch_index", -1)
        if idx < 0 or idx >= len(train_it):
            idx = random.randint(0, len(train_it))
        inputs = []
        outputs = []
        print(f"Generating {n_iterations} versions of sample {idx}", flush=True)
        for i in range(n_iterations):
            input, output = train_it[idx]
            inputs.append(input)
            if not input_only:
                outputs.append(output)
            print(f"{i + 1}/{n_iterations}", flush=True)
        input = np.stack(inputs, 0)
        transpose_axis = [4, 0, 1, 2, 3]
        input = np.transpose(input, transpose_axis)
        if not input_only:
            outputs = transpose_list(outputs) # (n_it, n out) -> (n_out, n_it)
            outputs = [np.transpose(np.stack(o, 0), transpose_axis) for o in outputs]
            output_name = ["EDM", "GCDM"]
        print(f"writing {len(outputs)+1} x {input.shape} to file: {file_path}", flush=True)
        with h5py.File(file_path, mode='w') as h5pyFile :
            h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/input", data=input)
            if not input_only:
                for i, o in enumerate(outputs):
                    h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/output_{i}_{output_name[i]}", data=o)
    else:
        # init model
        print("init model...")
        model = init_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(LR))
        # perform training
        train_it._close_datasetIO()
        if test_it is not None:
            test_it._close_datasetIO()
        checkpoint = tf.keras.callbacks.ModelCheckpoint(WEIGHT_PATH, monitor='val_loss' if test_it is not None else 'loss', verbose=1, save_best_only=False, save_weights_only=True)
        lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(min_lr=MIN_LR, factor=0.5, patience=PATIENCE, verbose=1, min_delta=0.001, monitor='val_loss' if test_it is not None else 'loss')
        tensorboard_callback = tf.keras.callbacks.TensorBoard(LOG_PATH)
        callbacks = [lr_schedule, checkpoint, tf.keras.callbacks.TerminateOnNaN(), StopOnLR(MIN_LR)]
        if tensorboard_callback is not None:
            callbacks.append(tensorboard_callback)
        if N_EPOCHS > 0:
            if WORKERS > 1:
                enq = tf.keras.utils.OrderedEnqueuer(train_it, use_multiprocessing=True, shuffle=True)
                enq.start(workers=WORKERS, max_queue_size=max(3, WORKERS))
                gen = enq.get()
            else:
                gen = train_it
            print("start training... ", flush=True)
            model.fit(gen, epochs=N_EPOCHS, steps_per_epoch=STEP_NUMBER, validation_data=test_it, callbacks=callbacks, workers=1, use_multiprocessing=False)
            if WORKERS > 1:
                enq.stop()
        # export model
        model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)