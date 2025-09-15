import argparse
import copy
import os, resource
import shutil
import random
import numpy as np
import numpy.ma as ma
import tensorflow as tf
import h5py
from math import isnan
import skfmm
import edt
import warnings
from importlib.metadata import version

from scipy.ndimage import center_of_mass

from dataset_iterator.image_data_generator import get_image_data_generator, data_generator_to_channel_postprocessing_fun
from dataset_iterator.datasetIO import MemoryIO
from dataset_iterator import extract_tile_random_zoom_function
from dataset_iterator.utils import transpose_list
from dataset_iterator import MultiChannelIterator, TrackingIterator
from dataset_iterator.keras_callbacks import StopOnLR, EpsilonCosineDecayCallback, LogsCallback, SafeModelCheckpoint, ReduceLROnPlateau2
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF

from distnet_2d.data.center_edm import compute_edm
from distnet_2d.data.dydx_iterator import ARRAY_KEYWORDS
from distnet_2d.model.architectures import get_architecture
from distnet_2d.model.distnet_2d_seg import get_distnet_2d_seg
from distnet_2d.data.medoid import get_medoid
from distnet_2d.utils.helpers import flatten_list

from training_core import open_config_file, get_iterator, should_load_dataset_in_shm, get_shm_info, \
    get_input_channel_and_label, chain_pp_fun, get_category_class_weights

__VERSION__ = "1.1.2"

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--train_only", action="store_true", help="train but no export")
parser.add_argument("--export_only", action="store_true", help="skip model training")
parser.add_argument("--test_data_augmentation", action="store_true", help="generate and store example of augmented data")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
parser.add_argument("--n_epochs", type=int, help="number of training epochs")
parser.add_argument("--step_number", type=int, help="number of training steps per epoch")
parser.add_argument("--patience", type=int, help="patience for learning rate decrease during training")
parser.add_argument("--learning_rate", type=float, help="initial learning rate for training")
parser.add_argument("--min_learning_rate", type=float, help="minimal learning rate for training")
parser.add_argument("--strategy",default="",type=str,help="distributed training strategy: multiworker-slurm or mirrored. Leave empty for default behaviour (single replica)")

if __name__ == "__main__":
    args = parser.parse_args()

    # get parameters
    config = open_config_file(args.config_dir, args.test_data_augmentation)
    t_p = config["training_parameters"]
    model_name = t_p["model_name"] + (f"_{args.model_idx}" if args.model_idx is not None else "")
    WEIGHT_PATH = os.path.join(args.config_dir, t_p["weight_dir"],  model_name  + ".h5") if len(t_p["weight_dir"])>0 else os.path.join(args.config_dir,  model_name + ".h5")
    LOAD_WEIGHT_PATH = t_p["load_model_file"] if len(t_p.get("load_model_file", "")) > 0 else None
    LOG_PATH = os.path.join(args.config_dir, t_p["log_dir"], model_name ) if len(t_p["log_dir"])>0 else os.path.join(args.config_dir, model_name )
    SAVED_MODEL_PATH = os.path.join(args.export_dir if args.export_dir is not None else args.config_dir, model_name)
    N_EPOCHS = args.n_epochs if args.n_epochs is not None else t_p.get("n_epochs", 500)
    STEP_NUMBER = args.step_number if args.step_number is not None else t_p.get("step_number", 200)
    PATIENCE = args.patience if args.patience is not None else t_p.get("patience", 40)
    LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
    MIN_LR = args.min_learning_rate if args.min_learning_rate is not None else t_p.get("min_learning_rate", 5e-7)
    EPSILON_RANGE = t_p.get("epsilon_range", [1e-7, 1e-7])
    EPSILON_RANGE = [max(EPSILON_RANGE), min(EPSILON_RANGE)]
    if args.strategy == "multiworker-slurm":
        WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    else:
        WORKERS = t_p.get("multiprocessing_workers", 1)
    WORKERS = min(os.cpu_count(), WORKERS)
    USE_SHARED_MEM = t_p.get("use_shared_memory", False)
    SHUFFLE = not args.test_data_augmentation
    START_EPOCH = t_p.get("epoch_start", 0)

    #h5py._errors.silence_errors()

    warnings.filterwarnings("ignore")

    print(f"Script version: {__VERSION__}; dataset_iterator version: {version('dataset_iterator')}; DiSTNet2D version: {version('DiSTNet2D')}")
    print(f"configuration file found. ")


    def init_iterator(ds_conf, step_number, dataset=None, **kwargs):
        data_aug_params = ds_conf.get("data_augmentation", {})
        channel_names = ds_conf.get("channel_name", "raw")
        if not isinstance(channel_names, (list, tuple)):
            channel_names = [channel_names]
        elif isinstance(channel_names, tuple):
            channel_names = list(channel_names)
        channel_names = [f"/{cn}" if cn[0] != "/" else cn for cn in channel_names]
        label_names = ds_conf.get("label_name", [])
        if not isinstance(label_names, (list, tuple)):
            label_names = [label_names]
        elif isinstance(label_names, tuple):
            label_names = list(label_names)
        label_names = [f"/{cn}" if cn[0] != "/" else cn for cn in label_names]
        category_number = config["model_architecture"].get("category_number", 0)
        if dataset is None:
            dataset = ds_conf["path"]
            memory_persistent = WORKERS > 1 and not args.test_data_augmentation and should_load_dataset_in_shm(dataset, mode=ds_conf.get("shared_memory", "auto"))
        else:
            memory_persistent = isinstance(dataset, MemoryIO)
        batch_size = ds_conf["batch_size"]
        if "tiling_parameters" in ds_conf:
            tiling_parameters = ds_conf["tiling_parameters"]
            extract_tiles_fun = extract_tile_random_zoom_function(**tiling_parameters)
        else:
            extract_tiles_fun = None
        scaling_parameters = data_aug_params.get("scaling_parameters", {})
        if not isinstance(scaling_parameters, (list, tuple)):
            scaling_parameters = [scaling_parameters]
        for sp, cname in zip(scaling_parameters, channel_names):
            sp["dataset"] = dataset
            sp["channel_name"] = cname
        affine_transform_parameters = data_aug_params.get("affine_transform_parameters", None)
        data_generators = [get_image_data_generator(scaling_parameters=sp, affine_transform_parameters=affine_transform_parameters) for sp in scaling_parameters]
        data_gen_input_label = [ get_image_data_generator(scaling_parameters=[], affine_transform_parameters=affine_transform_parameters) ] * len(label_names)
        affine_transform_parameters_mask = None if affine_transform_parameters is None else {**affine_transform_parameters, "interpolation_order": 0}
        mask_generator = get_image_data_generator(scaling_parameters=[], affine_transform_parameters=affine_transform_parameters_mask)
        # perform illumination at the end: after elastic deform
        illumination_parameters = copy.deepcopy(data_aug_params.get("illumination_transform",  [data_aug_params.get("illumination_parameters", None)]))
        ill_fun_list = []
        for cidx, ip in enumerate(illumination_parameters):
            if ip is not None and ip.pop("mode", True):
                illumination_gen = get_image_data_generator(illumination_parameters=ip)
                ill_fun_list.append( data_generator_to_channel_postprocessing_fun(illumination_gen,[0 if cidx ==0 else cidx + 1])) # channel #1 is reserved to labels
        illu_fun = chain_pp_fun(ill_fun_list) if len(ill_fun_list) > 0 else None
        exclude_void = ds_conf.get("exclude_empty_frames", False)
        seg_args = config.get("segmentation", {})
        use_gdcm = seg_args.get("center_distance_mode", "GEODESIC") == "GEODESIC"
        center_mode = seg_args.get("center_mode", "MEDOID")
        scale_edm = seg_args.get("scale_edm", False)
        assert center_mode in ["MEDOID", "GEOMETRICAL"], f"Invalid center mode = {center_mode} should be either MEDOID or GEOMETRICAL"

        def edm_fun(labels):
            edm = edt.edt(labels, black_border=False)
            edm[labels==0] = -1
            return edm

        def edm_fun_out(labels):
            edm = edm_fun(labels)
            if scale_edm:
                all_labels = np.unique(labels)
                all_labels = [l for l in all_labels if l != 0]
                for l in all_labels:
                    if l!=0:
                        mask = labels == l
                        edm_masked = edm[mask]
                        edm[mask] = edm_masked / np.max(edm_masked)
            return edm

        def get_centers(labels):
            all_labels = np.unique(labels)
            all_labels = [int(round(l)) for l in all_labels if l != 0]
            if center_mode == "MEDOID":
                return [get_medoid(*np.where(labels == l)) for l in all_labels]
            else:
                return center_of_mass(labels, labels, all_labels)

        def gcdm_fun(labels):
            centers = get_centers(labels)
            count = 0
            m = np.ones_like(labels)
            for center in centers:
                if not (isnan(center[0]) or isnan(center[1])):
                    m[int(round(center[0])), int(round(center[1]))] = 0
                    count += 1
            if count > 0:
                m = ma.masked_array(m, ~labels.astype(bool))
                return skfmm.distance(m).astype(np.float32)
            else:
                return np.zeros_like(labels, dtype=np.float32)

        def ecdm_fun(labels):
            centers = get_centers(labels)
            edcm = np.zeros_like(labels, dtype=np.float32)
            compute_edm(centers, edcm)
            return edcm

        def apply_batchwise(fun):
            def result_fun(batch):
                images = [fun(batch[b,...,0]) for b in range(batch.shape[0])]
                batch_res = np.stack(images, 0)
                return np.expand_dims(batch_res, -1)
            return result_fun

        label_idx = len(channel_names) + 1
        label_cidx = [label_idx + ci for ci in range(len(label_names))]
        cat_idx = len(channel_names) + 1 + len(label_names)

        def pp_fun(batch_by_channel):
            if illu_fun is not None:
                illu_fun(batch_by_channel)
            if category_number > 1:
                categoryArray = batch_by_channel['arrays'][0]
                labelIm = batch_by_channel[1]
                catIm = np.zeros_like(labelIm)
                for b in range(labelIm.shape[0]):
                    lIm = labelIm[b]
                    cIm = catIm[b]
                    cA = categoryArray[b]
                    if len(cA.shape) == 3:
                        cA = cA[:,0,0]
                    elif len(cA.shape) == 2:
                        cA = cA[:,0]
                    all_labels = np.unique(lIm)
                    for l in all_labels:
                        if l != 0:
                            cIm[ lIm == l ] = cA[int(round(l)) - 1] + 1
                batch_by_channel[cat_idx] = catIm
        print(f"channels : {[channel_names[0], '/regionLabels'] + channel_names[1:] + label_names} inputs: {[0] + [1 + ci for ci in range(1, len(channel_names))] + [ci for ci in label_cidx for _ in range(2)]} outputs: {[1, 1, cat_idx] if category_number > 1 else [1, 1]} mask: {[1] + label_cidx}")
        iterator_params = dict(dataset=dataset,
                               channel_keywords=[channel_names[0], '/regionLabels'] + channel_names[1:] + label_names,
                               array_keywords = [ARRAY_KEYWORDS[1]] if category_number > 1 else None,
                               group_keyword=ds_conf.get("keyword", None),
                               input_channels=[0] + [1 + ci for ci in range(1, len(channel_names))] + [ci for ci in label_cidx for _ in range(2)], # edm / cdm
                               output_channels=[1, 1, cat_idx] if category_number > 1 else [1, 1], # cat_idx = placeholder for category computed in post_processing fun
                               mask_channels=[1] + label_cidx,
                               batch_size=batch_size, step_number=step_number,
                               extract_tile_function=extract_tiles_fun, shuffle=kwargs.get("shuffle", True),
                               image_data_generators=[data_generators[0], mask_generator] + data_generators[1:] + data_gen_input_label,
                               elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None),
                               channels_postprocessing_function=pp_fun,
                               input_postprocessing_functions = [None]*len(channel_names) + [apply_batchwise(edm_fun) if idx==0 else apply_batchwise(gcdm_fun) for _ in label_cidx for idx in range(2)],
                               output_postprocessing_functions=[apply_batchwise(edm_fun_out), apply_batchwise(gcdm_fun if use_gdcm else ecdm_fun)] + ([None] if category_number > 1 else []),
                               void_mask_proportion=[0, 0] if exclude_void else None,
                               memory_persistent=memory_persistent)
        return MultiChannelIterator(**iterator_params)


    def init_model():
        arch_args = config["model_architecture"].copy()
        skip_connections = arch_args.pop("skip_connections", True)
        nchan, nlabel = get_input_channel_and_label(config)
        n_inputs = nchan + 2 * nlabel # for each label: edm and gcdm are computed
        if arch_args.get("architecture_type", "blend").lower() == "enc_dec":
            arch_args["architecture_type"] = "blend" # enc_dec is similar to distnet2d blend architecture, wihtout the blending part
        shape = config["dataset_parameters"]["input_shape"]
        input_shape = [None if s <= 0 else s for s in shape]
        arch_args["spatial_dimensions"] = input_shape
        category_number = arch_args.pop("category_number", 0)
        category_class_weights = get_category_class_weights(config, category_number, category_keyword=ARRAY_KEYWORDS[1], max_weight=10) if category_number > 1 else None
        if category_class_weights is not None:
            print(f"Category class weights: {category_class_weights}")
        arch = get_architecture(arch_args.pop("architecture_type", "blend"), **arch_args)
        seg_args = config.get("segmentation", {})
        cdm_loss_radius = seg_args.get("cdm_loss_radius", 0)
        model = get_distnet_2d_seg(n_inputs=n_inputs, config=arch, skip_connections=skip_connections, shared_encoder=False, accum_steps=1, l2_reg=0, scale_edm = seg_args.get("scale_edm", False), category_number=category_number, category_class_weights=category_class_weights, cdm_loss_radius=cdm_loss_radius)
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
        tf.saved_model.save(model, SAVED_MODEL_PATH)
        print("model saved", flush=True)
    else:
        print(f"init iterator...", flush=True)
        test_param = config.get("test_data_augmentation_parameters", {})
        if args.test_data_augmentation and "frame_subsampling" in test_param:
            for ds_params in config["dataset_list"]:
                ds_params["data_augmentation"]["frame_subsampling"] = test_param["frame_subsampling"]

        if args.test_data_augmentation:
            train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE)
            test_param = config.get("test_data_augmentation_parameters", {})
            input_only = test_param.get("input_only", True)
            n_iterations = test_param.get("iteration_number", 10)
            root_path = "/dataTemp" if os.path.exists("/dataTemp") else "/data"
            file_path = os.path.join(root_path, "test_data_augmentation.h5")
            idx = test_param.get("batch_index", -1)
            if idx < 0 or idx >= len(train_it):
                idx = random.randint(0, len(train_it)-1)
            inputs = []
            outputs = []
            print(f"Generating {n_iterations} versions of sample {idx}", flush=True)
            for i in range(n_iterations):
                input, output = train_it[idx]
                inputs.append(input)
                if not input_only:
                    outputs.append(output)
                print(f"{i + 1}/{n_iterations}", flush=True)
            transpose_axis = [4, 0, 1, 2, 3]
            cnames, lnames = get_input_channel_and_label(config, return_names=True)
            if isinstance(inputs[0], (list, tuple)):
                inputs = transpose_list(inputs)  # (n_it, n_in) -> (n_in, n_it)
                inputs = [np.transpose(np.stack(i, 0), transpose_axis) for i in inputs]
                input_names = cnames + [f"{ln}_{'EDM' if i==0 else 'CDM'}" for ln in lnames for i in range(2)]
            else:
                inputs = np.stack(inputs, 0)
                inputs = [np.transpose(inputs, transpose_axis)]
                input_names = cnames

            if not input_only:
                outputs = transpose_list(outputs) # (n_it, n out) -> (n_out, n_it)
                outputs = [np.transpose(np.stack(o, 0), transpose_axis) for o in outputs]
                output_names = ["EDM", "CDM"] if len(outputs)==2 else ["EDM", "CDM", "Category"]

            print(f"writing {len(outputs)+len(inputs)} x {inputs[0].shape} to file: {file_path}", flush=True)
            with h5py.File(file_path, mode='w') as h5pyFile :
                for i, o in enumerate(inputs):
                    h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/input_{i}_{input_names[i]}", data=o)
                if not input_only:
                    for i, o in enumerate(outputs):
                        h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/output_{i}_{output_names[i]}", data=o)
        else:
            # Define the training distribution strategy
            if args.strategy == "multiworker-slurm":
                # build multi-worker environment from Slurm variables
                cluster_resolver = tf.distribute.cluster_resolver.SlurmClusterResolver(
                    port_base=15000
                )

                # use NCCL communication protocol
                implementation = (
                    tf.distribute.experimental.CommunicationImplementation.NCCL
                )
                communication_options = tf.distribute.experimental.CommunicationOptions(
                    timeout_seconds=0.0,
                    implementation=implementation,
                )

                # declare distribution strategy
                strategy = tf.distribute.MultiWorkerMirroredStrategy(
                    cluster_resolver=cluster_resolver,
                    communication_options=communication_options,
                )

            elif args.strategy == "mirrored":
                strategy = tf.distribute.MirroredStrategy()
            else:
                strategy = (
                    tf.distribute.get_strategy()
                )  # will return a default single replica strategy

            # init model
            print("init model...")
            with strategy.scope():
                model = init_model()
                model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON_RANGE[0]))
            # perform training
            test_it = None
            checkpoint = SafeModelCheckpoint(WEIGHT_PATH, monitor='val_loss' if test_it is not None else 'loss', verbose=1, save_best_only=False, save_weights_only=True)
            lr_schedule = ReduceLROnPlateau2(min_lr=MIN_LR, factor=0.5, patience=PATIENCE, verbose=1, min_delta=0.001, monitor='val_loss' if test_it is not None else 'loss')
            tensorboard_callback = None # tf.keras.callbacks.TensorBoard(LOG_PATH)
            callbacks = [lr_schedule, checkpoint, tf.keras.callbacks.TerminateOnNaN(), StopOnLR(MIN_LR)]
            if tensorboard_callback is not None:
                callbacks.append(tensorboard_callback)
            log_cb = LogsCallback(LOG_PATH + ".csv", start_epoch=START_EPOCH)
            callbacks.append(log_cb)
            if EPSILON_RANGE[1]!=EPSILON_RANGE[0]:
                eps_schedule = EpsilonCosineDecayCallback(decay_steps=N_EPOCHS * STEP_NUMBER, start_epsilon=EPSILON_RANGE[0],  min_epsilon=EPSILON_RANGE[1], start_step=START_EPOCH * STEP_NUMBER, verbose=1)
                callbacks.append(eps_schedule)

            N_EPOCHS -= START_EPOCH
            if N_EPOCHS > 0:
                train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=True)
                if WORKERS > 1:
                    # check available shm:
                    shm = get_shm_info()
                    if shm is not None and shm[2] < 1:
                        print( f"Warning: available shared memory is low: {shm[2]:.2f}/{shm[0]:.2f}G, this can hamper multiprocessing", force=True)
                    #enq = tf.keras.utils.OrderedEnqueuer(train_it, use_multiprocessing=True, shuffle=True)
                    enq = OrderedEnqueuerCF(train_it, shuffle=True)
                    enq.start(workers=WORKERS, max_queue_size=max(3, min(STEP_NUMBER, WORKERS)))
                    gen = enq.get()
                else:
                    gen = train_it
                print("start training... ", flush=True)
                model.fit(gen, epochs=N_EPOCHS, steps_per_epoch=STEP_NUMBER, validation_data=test_it, callbacks=callbacks, workers=1, use_multiprocessing=False)
                if WORKERS > 1:
                    enq.stop()
                print("training successful", flush=True)
            elif START_EPOCH > 0:
                print("Start Epoch is greater than Epoch number.", flush=True)
            train_it.close()

            if not args.train_only: # export model
                print("saving model...", flush=True)
                if args.strategy == "multiworker-slurm":
                    is_chief = (
                        cluster_resolver.task_type == "worker"
                        and cluster_resolver.task_id == 0
                    )

                    save_path = (
                        SAVED_MODEL_PATH
                        if is_chief
                        else SAVED_MODEL_PATH + "_tmp_" + os.environ.get("SLURM_PROCID", "")
                    )

                    model.save(
                        save_path,
                        include_optimizer=False,
                        save_traces=True,
                        inference=True,
                    )
                    print("model saved", flush=True)

                    if not is_chief:
                        print(f"cleaning temp models at {save_path}", flush=True)
                        shutil.rmtree(save_path)  # clean up for non chief worker
                else:
                    model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True)
                    # model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)
                    print("model saved", flush=True)
                    #tf.saved_model.save(model, SAVED_MODEL_PATH)
                    #print("model saved", flush=True)
