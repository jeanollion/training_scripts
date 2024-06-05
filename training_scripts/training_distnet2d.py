import argparse
import os
import random
import numpy as np
import tensorflow as tf
import h5py
import copy
from importlib.metadata import version
from dataset_iterator.image_data_generator import get_image_data_generator, data_generator_to_channel_postprocessing_fun
from dataset_iterator.datasetIO import MemoryIO
from dataset_iterator import extract_tile_random_zoom_function, ConcatIterator
from dataset_iterator.utils import transpose_list
from dataset_iterator.hard_sample_mining import HardSampleMiningCallback, compute_metrics
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF
from dataset_iterator.keras_callbacks import StopOnLR, EpsilonCosineDecayCallback, LogsCallback, SafeModelCheckpoint, ReduceLROnPlateau2
from distnet_2d.data import DyDxIterator
from distnet_2d.data.swim1d import get_swim1d_function
from distnet_2d.model.architectures import get_architecture
from distnet_2d.model.distnet_2d import get_distnet_2d
from distnet_2d.utils.metrics_tf import get_metrics_fun
from training_core import open_config_file, get_iterator, chain_pp_fun, set_to_iterator, should_load_dataset_in_shm, get_shm_info, get_shm_nfiles

__VERSION__ = '1.0.4'
parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--load_model_idx", type=int, help="index of model to load weights from")
parser.add_argument("--export_only", action="store_true", help="skip model training")
parser.add_argument("--test_data_augmentation", action="store_true", help="generate and store example of augmented data")
parser.add_argument("--compute_metrics", action="store_true", help="compute loss")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
parser.add_argument("--n_epochs", type=int, help="number of training epochs")
parser.add_argument("--step_number", type=int, help="number of training steps per epoch")
parser.add_argument("--patience", type=int, help="patience for learning rate decrease during training")
parser.add_argument("--learning_rate", type=float, help="initial learning rate for training")
parser.add_argument("--min_learning_rate", type=float, help="minimal learning rate for training")
args = parser.parse_args()

# get parameters
config = open_config_file(args.config_dir, args.test_data_augmentation)
t_p = config["training_parameters"]
model_name = t_p["model_name"] + (f"_{args.model_idx}" if args.model_idx is not None else "")
load_model_filename = t_p["load_model_filename"] + (f"_{args.load_model_idx}" if args.load_model_idx is not None else "") if len(t_p.get("load_model_filename", "")) > 0 else None
WEIGHT_PATH = os.path.join(args.config_dir, t_p["weight_dir"],  model_name + ".h5") if len(t_p["weight_dir"])>0 else os.path.join(args.config_dir,  model_name + ".h5")
LOAD_WEIGHT_PATH = (os.path.join(args.config_dir, t_p["weight_dir"], load_model_filename) if len(t_p["weight_dir"]) > 0 else os.path.join(args.config_dir, load_model_filename)) if load_model_filename is not None else None
LOG_PATH = os.path.join(args.config_dir, t_p["log_dir"], model_name ) if len(t_p["log_dir"])>0 else os.path.join(args.config_dir, model_name )
SAVED_MODEL_PATH = os.path.join(args.export_dir if args.export_dir is not None else args.config_dir, model_name)
N_EPOCHS = args.n_epochs if args.n_epochs is not None else t_p.get("n_epochs", 500)
STEP_NUMBER = args.step_number if args.step_number is not None else t_p.get("step_number", 200)
PATIENCE = args.patience if args.patience is not None else t_p.get("patience", 40)
LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
MIN_LR = args.min_learning_rate if args.min_learning_rate is not None else t_p.get("min_learning_rate", 5e-7)
EPSILON_RANGE = t_p.get("epsilon_range", [1e-7, 1e-7])
EPSILON_RANGE = [max(EPSILON_RANGE), min(EPSILON_RANGE)]
WORKERS = min(os.cpu_count(), t_p.get("multiprocessing_workers", 1))
SHUFFLE = not args.test_data_augmentation
START_EPOCH = t_p.get("start_epoch", 0)

print(f"Script version: {__VERSION__}; dataset_iterator version: {version('dataset_iterator')}; DiSTNet2D version: {version('DiSTNet2D')}")
print(f"configuration file found. ")

# TEST VARIABLES TO ADD TO CONFIGURATION IF RELEVANT
PREDICT_EDM_DERIVATIVES = False
PREDICT_GCDM_DERIVATIVES = False
EDM_DERIVATIVE_LOSS = False
GCDM_DERIVATIVE_LOSS = False


def init_iterator(step_number, shuffle, dataset=None, **ds_kwargs):
    data_aug_params = ds_kwargs.get("data_augmentation", {})
    dataset_features = ds_kwargs.get("dataset_features", {})
    arch_params = config["model_architecture"]
    channel_name = ds_kwargs.get("channel_name", "raw")
    if dataset is None:
        dataset = ds_kwargs["path"]
        memory_persistent = WORKERS > 1 and not args.test_data_augmentation and should_load_dataset_in_shm(dataset, mode=ds_kwargs.get("shared_memory", "auto"))
    else:
        memory_persistent = isinstance(dataset, MemoryIO)
    batch_size = ds_kwargs["batch_size"]
    if "tiling_parameters" in ds_kwargs:
        tiling_parameters = ds_kwargs["tiling_parameters"]
        extract_tiles_fun = extract_tile_random_zoom_function(**tiling_parameters)
    else:
        extract_tiles_fun = None
    scaling_parameters = data_aug_params.get("scaling_parameters", {})
    scaling_parameters["dataset"] = dataset
    scaling_parameters["channel_name"] = channel_name
    affine_transform_parameters = data_aug_params.get("affine_transform_parameters", None)
    data_generator = get_image_data_generator(scaling_parameters=scaling_parameters, affine_transform_parameters=affine_transform_parameters)
    affine_transform_parameters_mask = None if affine_transform_parameters is None else {**affine_transform_parameters, "interpolation_order": 0}
    mask_generator = get_image_data_generator(scaling_parameters=[], affine_transform_parameters=affine_transform_parameters_mask)
    pp_fun_list = []
    swim1D_params = data_aug_params.get("swim1d_parameters", None)
    if swim1D_params is not None:
        pp_fun_list.append(get_swim1d_function(1, swim1D_params.get("distance", 50), swim1D_params.get("min_gap", 3), swim1D_params.get("closed_end", True)))
    illumination_parameters = data_aug_params.get("illumination_parameters", None)
    if illumination_parameters is not None: # perform illumination at the end: after elastic deform and swim
        illumination_gen = get_image_data_generator(illumination_parameters=illumination_parameters)
        pp_fun_list.append(data_generator_to_channel_postprocessing_fun(illumination_gen, [0]))

    pp_fun = chain_pp_fun(pp_fun_list)
    iterator_params = dict(erase_edge_cell_size=data_aug_params.get("erase_edge_cell_size", 0),
                           aug_remove_prob=data_aug_params.get("static_probability", 0.01),
                           next=arch_params.get("next", True),
                           center_mode=dataset_features.get("center_mode", "MEDOID"),
                           frame_window=arch_params.get("frame_window", 3),
                           image_data_generators=[data_generator, mask_generator],
                           elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None),
                           channels_postprocessing_function=pp_fun, verbose=False and args.test_data_augmentation, memory_persistent=memory_persistent)
    return DyDxIterator(dataset=dataset, channel_keywords=[channel_name, '/regionLabels'], group_keyword=ds_kwargs.get("keyword", None),
                        batch_size=batch_size, step_number=step_number, extract_tile_function=extract_tiles_fun, return_edm_derivatives=EDM_DERIVATIVE_LOSS or PREDICT_EDM_DERIVATIVES,
                        aug_frame_subsampling=data_aug_params.get("frame_subsampling", 1), shuffle=shuffle,
                        **iterator_params)


def init_model():
    arch_args = copy.deepcopy(config["model_architecture"])
    frame_window = arch_args.pop("frame_window", 3)
    next = arch_args.pop("next", True)
    shape = config["dataset_parameters"]["input_shape"]
    input_shape = [None if s <= 0 else s for s in shape]
    arch_args["spatial_dimensions"] = input_shape
    arch = get_architecture(arch_args.pop("architecture_type", "blend"), **arch_args)
    model = get_distnet_2d(input_shape, config=arch, next=next, frame_window=frame_window, accum_steps=1, l2_reg=0, predict_edm_derivatives=PREDICT_EDM_DERIVATIVES, predict_gcdm_derivatives=PREDICT_GCDM_DERIVATIVES, edm_derivative_loss=EDM_DERIVATIVE_LOSS, gcdm_derivative_loss=GCDM_DERIVATIVE_LOSS)
    if args.export_only or args.compute_metrics and os.path.exists(WEIGHT_PATH):
        assert os.path.exists(WEIGHT_PATH), f"weights {WEIGHT_PATH} not found"
        model.load_weights(WEIGHT_PATH)
        print(f"Weights loaded : {WEIGHT_PATH}", flush=True)
    elif LOAD_WEIGHT_PATH is not None or args.compute_metrics:
        assert os.path.exists(LOAD_WEIGHT_PATH), f"weights {LOAD_WEIGHT_PATH} not found"
        if os.path.isdir(LOAD_WEIGHT_PATH):
            loaded_model = tf.keras.models.load_model(LOAD_WEIGHT_PATH)
            model.set_weights(loaded_model.get_weights())
        else:
            model.load_weights(LOAD_WEIGHT_PATH)
        print(f"Weights loaded : {LOAD_WEIGHT_PATH}", flush=True)
    return model


def configure_metrics_iterator(iterator):
    def fun(it):
        it.return_central_only=True
        it.incomplete_last_batch_mode = 0
        it.return_label_rank = True
    set_to_iterator(iterator, fun)


def metrics_fun(center_scale, frame_window, long_range:bool=True):
    metrics_fun_ = get_metrics_fun(center_scale=center_scale)
    def fun(y_true, y_pred):
        fw = frame_window
        n_frame_pairs = fw * 2
        if long_range:
            n_frame_pairs += (fw - 1) * 2
        d_indices = [fw - 1, n_frame_pairs + fw]
        lm_indices = [d_indices[0] * 3 + i for i in range(3)] + [d_indices[1] * 3 + i for i in range(3)]
        return metrics_fun_(y_pred[0][..., fw:fw + 1], y_pred[1][..., fw:fw + 1],
                            tf.gather(y_pred[2], indices=d_indices, axis=-1),
                            tf.gather(y_pred[3], indices=d_indices, axis=-1),
                            tf.gather(y_pred[4], indices=lm_indices, axis=-1), y_true[0], y_true[2], y_true[3],
                            y_true[4], y_true[5], y_true[6], y_true[7])
    return fun


if args.export_only:
    print(f"export only: init model with weights: {WEIGHT_PATH} (exist: {os.path.exists(WEIGHT_PATH)})", flush=True)
    model = init_model()
    # export model
    model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)
    print("model saved", flush=True)
else:
    print(f"init iterator...", flush=True)
    test_param = config.get("test_data_augmentation_parameters", {})
    if args.test_data_augmentation and "frame_subsampling" in test_param:
        for ds_params in config["dataset_list"]:
            ds_params["data_augmentation"]["frame_subsampling"] = test_param["frame_subsampling"]
    #it_steps = STEP_NUMBER if WORKERS==1 else 0

    test_it = None
    if args.test_data_augmentation:
        train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE)
        test_param = config.get("test_data_augmentation_parameters", {})
        input_only = test_param.get("input_only", True)
        n_iterations = test_param.get("iteration_number", 10)
        root_path = "/dataTemp" if os.path.exists("/dataTemp") else "/data"
        file_path = os.path.join(root_path, "test_data_augmentation.h5")
        idx = test_param.get("batch_index", -1)
        if idx < 0 or idx >= len(train_it):
            idx = random.randint(0, len(train_it))
        inputs = []
        outputs = []
        print(f"Generating {n_iterations} versions of sample {idx}", flush=True)
        for i in range(n_iterations):
            input, output = train_it[idx]
            #idx_a = np.copy(train_it.index_array)
            #print(f"index array: {idx_a}")
            #if isinstance(train_it, ConcatIterator):
            #    index_it = train_it._get_it_idx(idx_a)
            #    print(f"it {index_it[idx]} shuffle: {train_it.iterators[index_it[idx]].shuffle} index array: {train_it.iterators[index_it[idx]].index_array}")
            inputs.append(input)
            if not input_only:
                outputs.append(output)
            print(f"{i + 1}/{n_iterations}", flush=True)
        train_it.close()
        input = np.stack(inputs, 0)
        transpose_axis = [4, 0, 1, 2, 3]
        input = np.transpose(input, transpose_axis)
        if not input_only:
            outputs = transpose_list(outputs) # (n_it, n out) -> (n_out, n_it)
            outputs = [np.transpose(np.stack(o, 0), transpose_axis) for o in outputs]
            output_name = ["EDM", "GCDM", "dY", "dX", "Category"]
        print(f"writing {len(outputs)+1} x {input.shape} to file: {file_path}", flush=True)
        with h5py.File(file_path, mode='w') as h5pyFile :
            h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/input", data=input)
            if not input_only:
                for i, o in enumerate(outputs):
                    h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/output_{i}_{output_name[i]}", data=o)
    elif args.compute_metrics:
        model = init_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON_RANGE[0]))
        predict_fun = lambda x: model(x, training=False)
        hsm_it = get_iterator(config, init_iterator, step_number=0, shuffle=False)
        configure_metrics_iterator(hsm_it)
        hard_sample_mining_param = t_p.get("hard_sample_mining", {})
        center_scale = hard_sample_mining_param.get("center_scale", 4) if hard_sample_mining_param is not None else 4
        metrics = compute_metrics(hsm_it, predict_fun, metrics_fun(center_scale=center_scale, frame_window=config["model_architecture"].get("frame_window", 3)), disable_augmentation=True, disable_channel_postprocessing=True, verbose=2)
        root_path = "/dataTemp" if os.path.exists("/dataTemp") else "/data"
        path = os.path.join(root_path, "metrics.csv")
        print(f"saving metrics of shape: {metrics.shape} to path: {path}", flush=True)
        #print(f"shm n files: {get_shm_nfiles()}")
        np.savetxt(path, metrics, delimiter=";", header="IoU;CenterPosition;CenterValue;DisplacementL2;LinkMultiplicity")
    else: # training
        train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE)
        # init model
        print("init model...", flush=True)
        model = init_model()
        model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON_RANGE[0]))
        # perform training
        checkpoint = SafeModelCheckpoint(WEIGHT_PATH, monitor='val_loss' if test_it is not None else 'loss', verbose=1, save_best_only=False, save_weights_only=True)
        lr_schedule = ReduceLROnPlateau2(min_lr=MIN_LR, factor=0.5, patience=PATIENCE, verbose=1, min_delta=0.001, monitor='val_loss' if test_it is not None else 'loss')
        tensorboard_callback = None #tf.keras.callbacks.TensorBoard(LOG_PATH)
        callbacks = [lr_schedule, checkpoint, tf.keras.callbacks.TerminateOnNaN(), StopOnLR(MIN_LR)]
        if tensorboard_callback is not None:
            callbacks.append(tensorboard_callback)
        log_cb = LogsCallback(LOG_PATH + ".csv", start_epoch=START_EPOCH)
        callbacks.append(log_cb)
        if EPSILON_RANGE[1] != EPSILON_RANGE[0]:
            eps_schedule = EpsilonCosineDecayCallback(decay_steps=N_EPOCHS * STEP_NUMBER, start_epsilon=EPSILON_RANGE[0],  min_epsilon=EPSILON_RANGE[1], start_step=START_EPOCH * STEP_NUMBER, verbose=1)
            callbacks.append(eps_schedule)
        hard_sample_mining_param = t_p.get("hard_sample_mining", None)
        if hard_sample_mining_param is not None:
            predict_fun = lambda x: model(x, training=False)
            period = hard_sample_mining_param.get("period", 0.1)
            if period < 1:
                period = int(N_EPOCHS * period)
            center_scale = hard_sample_mining_param.get("center_scale", 4)
            start_from = hard_sample_mining_param.get("start_from_epoch", 0)
            print("init hsm iterator...", flush=True)
            hsm_it = get_iterator(config, init_iterator, existing_iterator=train_it, step_number=0, shuffle=False) # needs to be a different iterator as iterator.return_central_only
            configure_metrics_iterator(hsm_it)
            hsm_cb = HardSampleMiningCallback(hsm_it, train_it, predict_fun, metrics_fun(center_scale=center_scale, frame_window=config["model_architecture"].get("frame_window", 3)), period, start_epoch=START_EPOCH, start_from_epoch=start_from, enrich_factor=hard_sample_mining_param.get("enrich_factor", 100), quantile_max=hard_sample_mining_param.get("quantile_max", None), quantile_min=hard_sample_mining_param.get("quantile_min", None), disable_channel_postprocessing=True, verbose=2)
            callbacks.append(hsm_cb)

        else:
            hsm_it = None
            hsm_cb = None
        if N_EPOCHS > 0:
            if WORKERS > 1:
                # check available shm:
                shm = get_shm_info()
                if shm is not None and shm[2] < 1:
                    print( f"Warning: available shared memory is low: {shm[2]:.2f}/{shm[0]:.2f}G, this can hamper multiprocessing", force=True)
                #enq = tf.keras.utils.OrderedEnqueuer(train_it, use_multiprocessing=True, shuffle=True)
                enq = OrderedEnqueuerCF(train_it, shuffle=True, use_shm=False, use_shared_array=True)
                if hsm_cb is not None:
                    hsm_cb.set_enqueuer(enq)
                enq.start(workers=WORKERS, max_queue_size=max(2, min(STEP_NUMBER, WORKERS)))
                gen = enq.get()

            else:
                gen = train_it
            if hsm_cb is not None:
                hsm_cb.initialize()
            model.fit(gen, epochs=N_EPOCHS, steps_per_epoch=STEP_NUMBER, validation_data=test_it, callbacks=callbacks, workers=1, use_multiprocessing=False)
            if WORKERS > 1:
                print("stopping enqueuer...", flush=True)
                enq.stop()
            print("end of training", flush=True)
        train_it.close(force=True)
        if hsm_cb is not None:
            hsm_cb.close()
        # export model
        print("saving model...", flush=True)
        model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)
        print("model saved", flush=True)