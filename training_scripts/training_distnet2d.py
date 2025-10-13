import argparse
import os
import shutil
import platform
import random
import time
from collections import defaultdict
import numpy as np
import tensorflow as tf
import h5py
import copy
from importlib.metadata import version
from dataset_iterator.image_data_generator import get_image_data_generator, data_generator_to_channel_postprocessing_fun
from dataset_iterator.datasetIO import MemoryIO, get_datasetIO
from dataset_iterator import extract_tile_random_zoom_function, ConcatIterator
from dataset_iterator.utils import transpose_list
from dataset_iterator.hard_sample_mining import HardSampleMiningCallback, compute_metrics
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF
from dataset_iterator.keras_callbacks import StopOnLR, EpsilonCosineDecayCallback, LogsCallback, SafeModelCheckpoint, ReduceLROnPlateau2
from distnet_2d.data import DyDxIterator
from distnet_2d.data.dydx_iterator import ARRAY_KEYWORDS
from distnet_2d.data.swim1d import get_swim1d_function
from distnet_2d.model.architectures import get_architecture
from distnet_2d.model.distnet_2d import get_distnet_2d
from distnet_2d.utils.helpers import get_background_foreground_counts, count_links
from distnet_2d.utils.metrics_tf import get_metrics_fun
from training_core import open_config_file, get_iterator, chain_pp_fun, set_to_iterator, should_load_dataset_in_shm, \
    get_shm_info, get_input_channel_and_label, get_category_class_weights, compute_category_weights

__VERSION__ = '1.1.3'
parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--train_only", action="store_true", help="train but no export")
parser.add_argument("--export_only", action="store_true", help="skip model training")
parser.add_argument("--test_data_augmentation", action="store_true", help="generate and store example of augmented data")
parser.add_argument("--compute_metrics", action="store_true", help="compute loss")
parser.add_argument("--test_predict", action="store_true", help="make predictions on evaluation dataset")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
parser.add_argument("--n_epochs", type=int, help="number of training epochs")
parser.add_argument("--step_number", type=int, help="number of training steps per epoch")
parser.add_argument("--patience", type=int, help="patience for learning rate decrease during training")
parser.add_argument("--learning_rate", type=float, help="initial learning rate for training")
parser.add_argument("--min_learning_rate", type=float, help="minimal learning rate for training")
parser.add_argument("--strategy",default="",type=str,help="distributed training strategy: multiworker-slurm or mirrored. Leave empty for default behaviour (single replica)")

if __name__ == "__main__":
    args = parser.parse_args()
    RUN_TEST = args.test_data_augmentation or args.test_predict
    # get parameters
    config = open_config_file(args.config_dir, RUN_TEST)
    t_p = config["training_parameters"]
    model_name = t_p["model_name"] + (f"_{args.model_idx}" if args.model_idx is not None else "")
    WEIGHT_PATH = os.path.join(args.config_dir,  model_name + ".h5")
    LOAD_WEIGHT_PATH = t_p["load_model_file"] if len(t_p.get("load_model_file", "")) > 0 else None
    LOG_PATH = os.path.join(args.config_dir, model_name )
    SAVED_MODEL_PATH = os.path.join(args.export_dir if args.export_dir is not None else args.config_dir, model_name)
    N_EPOCHS = args.n_epochs if args.n_epochs is not None else t_p.get("n_epochs", 500)
    STEP_NUMBER = args.step_number if args.step_number is not None else t_p.get("step_number", 200)
    PATIENCE = args.patience if args.patience is not None else t_p.get("patience", 40)
    LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
    MIN_LR = args.min_learning_rate if args.min_learning_rate is not None else t_p.get("min_learning_rate", 5e-7)
    EPSILON = 1e-7

    if args.strategy == "multiworker-slurm":
        WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    else:
        WORKERS = t_p.get("multiprocessing_workers", 1)
    WORKERS = min(os.cpu_count(), WORKERS)
    SHUFFLE = not RUN_TEST
    START_EPOCH = t_p.get("start_epoch", 0)
    print(f"Script version: {__VERSION__}; dataset_iterator version: {version('dataset_iterator')}; DiSTNet2D version: {version('DiSTNet2D')} python: {platform.python_version()}")
    print(f"configuration file found. ")

    def init_iterator(ds_conf, step_number, dataset=None, **kwargs):
        data_aug_params = ds_conf.get("data_augmentation", {})
        seg_args = config.get("segmentation", {})
        tracking = not seg_args.get("segment_only", False)
        arch_params = config["model_architecture"]
        category_number = arch_params.get("category_number", 0)
        channel_names = ds_conf.get("channel_name", "raw")
        if not isinstance(channel_names, (list, tuple)):
            channel_names = [channel_names]
        elif isinstance(channel_names, tuple):
            channel_names = list(channel_names)
        channel_names = [ f"/{cn}" if cn[0]!="/" else cn for cn in channel_names ]
        label_names = ds_conf.get("label_name", [])
        if not isinstance(label_names, (list, tuple)):
            label_names = [label_names]
        elif isinstance(label_names, tuple):
            label_names = list(label_names)
        label_names = [f"/{cn}" if cn[0] != "/" else cn for cn in label_names]
        if dataset is None:
            dataset = ds_conf["path"]
            memory_persistent = WORKERS > 1 and not RUN_TEST and should_load_dataset_in_shm(dataset, mode=ds_conf.get("shared_memory", "auto"))
        else:
            memory_persistent = isinstance(dataset, MemoryIO)
        batch_size = ds_conf["batch_size"]
        if "tiling_parameters" in ds_conf:
            tiling_parameters = ds_conf["tiling_parameters"]
            if "anchor_point_mask_idx" in tiling_parameters and tiling_parameters["anchor_point_mask_idx"] is not None:
                anchor_point_mask_idx = tiling_parameters["anchor_point_mask_idx"]
                # channel order is : raw, label, *additional_channels, *additional_labels
                if anchor_point_mask_idx == 0:
                    anchor_point_mask_idx = 1
                else:
                    assert 0<=anchor_point_mask_idx-1<len(label_names), f"invalid anchor point idx. 0=target label, >0 = additional label. must be in range [0; {len(label_names)+1}]"
                    anchor_point_mask_idx = 1 + len(channel_names) + anchor_point_mask_idx - 1
                tiling_parameters["anchor_point_mask_idx"] = anchor_point_mask_idx
            extract_tiles_fun = extract_tile_random_zoom_function(**tiling_parameters)
        else:
            extract_tiles_fun = None
        scaling_parameters = data_aug_params.get("scaling_parameters", {})
        if not isinstance(scaling_parameters, (list, tuple)):
            scaling_parameters = [scaling_parameters]
        if len(scaling_parameters) == 1 and len(channel_names) > 1 :
            scaling_parameters = [copy.deepcopy(scaling_parameters[0]) for _ in range(len(channel_names))]
        for sp, cname in zip(scaling_parameters, channel_names):
            sp["dataset"] = dataset
            sp["channel_name"] = cname
        affine_transform_parameters = data_aug_params.get("affine_transform_parameters", None)
        data_generators = [get_image_data_generator(scaling_parameters=sp, affine_transform_parameters=affine_transform_parameters) for sp in scaling_parameters]
        affine_transform_parameters_mask = None if affine_transform_parameters is None else {**affine_transform_parameters, "interpolation_order": 0}
        mask_generator = get_image_data_generator(scaling_parameters=[], affine_transform_parameters=affine_transform_parameters_mask)
        pp_fun_list = []
        swim1D_params = data_aug_params.get("swim1d_parameters", None)
        if swim1D_params is not None:
            pp_fun_list.append(get_swim1d_function(1, swim1D_params.get("distance", 50), swim1D_params.get("min_gap", 3), swim1D_params.get("closed_end", True)))
        # perform illumination at the end: after elastic deform and swim
        illumination_parameters = data_aug_params.get("illumination_transform", [data_aug_params.get("illumination_parameters", None)])
        for cidx, ip in enumerate(illumination_parameters):
            if ip is not None:
                pp_fun_list.append(data_generator_to_channel_postprocessing_fun(get_image_data_generator(illumination_parameters=ip), [0 if cidx ==0 else cidx + 1])) # channel #1 is reserved to labels

        pp_fun = chain_pp_fun(pp_fun_list)
        fw = arch_params["frame_window"]
        iterator_params = dict(erase_edge_cell_size=data_aug_params.get("erase_edge_cell_size", 0),
                               aug_remove_prob=data_aug_params.get("static_probability", 0.01),
                               next_frames=arch_params.get("next", True),
                               scale_edm = seg_args.get("scale_edm", False),
                               center_mode=seg_args.get("center_mode", "MEDOID"),
                               center_distance_mode=seg_args.get("center_distance_mode", "GEODESIC"),
                               frame_window=fw,
                               image_data_generators=[data_generators[0], mask_generator] + data_generators[1:],
                               elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None),
                               void_mask_proportion = [0, 0] if fw == 0 else None, # exclude empty frame, only when no frame window
                               channels_postprocessing_function=pp_fun, verbose=False and RUN_TEST, memory_persistent=memory_persistent)
        array_kw = (ARRAY_KEYWORDS[:1] if tracking else []) + (ARRAY_KEYWORDS[1:] if category_number>1 else [])
        return DyDxIterator(dataset=dataset, channel_keywords=[channel_names[0], '/regionLabels'] + channel_names[1:],
                            input_label_keywords=label_names, array_keywords=array_kw,
                            input_label_center_idx = seg_args.get("input_label_center_idx", -1),
                            tracking = tracking,
                            group_keyword=ds_conf.get("keyword", None),
                            batch_size=batch_size, step_number=step_number, extract_tile_function=extract_tiles_fun, return_edm_derivatives=seg_args.get("edm_derivatives", True),
                            aug_frame_subsampling=data_aug_params.get("frame_subsampling", 1), shuffle=kwargs.get("shuffle", True),
                            **iterator_params)


    def get_edm_class_weights(config: dict, max_weight:float):
        counts = np.array([0, 0], dtype="float128")
        for i, ds_conf in enumerate(config["dataset_list"]):
            counts += get_background_foreground_counts(ds_conf["path"], channel_keyword='/regionLabels', group_keyword=ds_conf.get("keyword", None))
        weights = compute_category_weights(dict(zip(["bck", "fore"], counts.tolist())), max_weight)
        return weights.astype("float32")

    def get_link_multiplicity_class_weights(config: dict, max_weight= 50):
        counts = {i: 0 for i in range(0, 3)}
        log = True
        for i, ds_conf in enumerate(config["dataset_list"]):
            dataset = get_datasetIO(ds_conf["path"], 'r')
            paths = dataset.get_dataset_paths(ARRAY_KEYWORDS[0], ds_conf.get("keyword", None))
            for p in paths:
                lm_array = dataset.get_dataset(p)
                s, m, n = count_links(lm_array, detailed = False)
                counts[0] += s
                counts[1] += m
                counts[2] += n
                log = False
            dataset.close()
        print(f"link multiplicity counts: {counts}")
        return compute_category_weights(counts, max_weight)


    def init_model(training:bool):
        arch_args = copy.deepcopy(config["model_architecture"])
        frame_window = arch_args.pop("frame_window", 3)
        next = arch_args.pop("next", True)
        inference_gap_number = arch_args.pop("inference_gap_number", 0)
        seg_args = config.get("segmentation", {})
        tracking = not seg_args.get("segment_only", False)
        shape = config["dataset_parameters"]["input_shape"]
        input_shape = [None if s <= 0 else s for s in shape]
        arch_args["spatial_dimensions"] = input_shape.copy()
        nchan, nlabel = get_input_channel_and_label(config)
        n_inputs = nchan + nlabel * 2 # for each label EDM and GDCM are added
        link_multiplicity_class_weights = get_link_multiplicity_class_weights(config, max_weight = 50) if training and tracking else None
        if training and tracking:
            print(f"link multiplicity weights: { {l:w for l,w in zip(['single', 'multiple', 'null'], link_multiplicity_class_weights)} }")
        category_number = arch_args.pop("category_number", 0)
        category_class_weights = get_category_class_weights(config, category_number, category_keyword=ARRAY_KEYWORDS[1], max_weight=10) if training and category_number > 1 else None
        if category_class_weights is not None:
            print(f"Category class weights: {category_class_weights}")
        edm_max_weight = seg_args.get("edm_max_frequency_weight", 0)
        edm_frequency_weights = get_edm_class_weights(config, edm_max_weight) if training and edm_max_weight>0 else None
        if edm_frequency_weights is not None:
            print(f"edm background/foreground balancing weights {edm_frequency_weights}")
        arch = get_architecture(arch_args.pop("architecture_type", "blend"), **arch_args)
        cdm_loss_radius = seg_args.get("cdm_loss_radius", 0)
        def make_model(legacy:bool=False):
            return get_distnet_2d(spatial_dimensions=input_shape, n_inputs=n_inputs, config=arch, next=next, frame_window=frame_window, tracking=tracking, accum_steps=1, l2_reg=0, edm_frequency_weights=edm_frequency_weights, edm_derivative_loss=seg_args.get("edm_derivatives", True), scale_edm = seg_args.get("scale_edm", False), cdm_derivative_loss=seg_args.get("cdm_derivatives", True), cdm_loss_radius=cdm_loss_radius, link_multiplicity_class_weights=link_multiplicity_class_weights, category_number=category_number, category_class_weights=category_class_weights, inference_gap_number=inference_gap_number, legacy_multi_input_arch = legacy)
        model = make_model()
        if args.export_only or ( (args.compute_metrics or args.test_predict) and os.path.exists(WEIGHT_PATH)):
            assert os.path.exists(WEIGHT_PATH), f"weights {WEIGHT_PATH} not found"
            try:
                model.load_weights(WEIGHT_PATH)
            except Exception as e: # re-try in legacy mode
                print(e)
                model = make_model(True)
                model.load_weights(WEIGHT_PATH)

            print(f"Weights loaded : {WEIGHT_PATH}", flush=True)
        elif LOAD_WEIGHT_PATH is not None or args.compute_metrics or args.test_predict :
            assert os.path.exists(LOAD_WEIGHT_PATH), f"weights {LOAD_WEIGHT_PATH} not found"
            if os.path.isdir(LOAD_WEIGHT_PATH):
                loaded_model = tf.keras.models.load_model(LOAD_WEIGHT_PATH)
                try:
                    model.set_weights(loaded_model.get_weights())
                except Exception as e: # re-try in legacy mode
                    print(e)
                    model = make_model(True)
                    model.set_weights(loaded_model.get_weights())
            else:
                try:
                    model.load_weights(LOAD_WEIGHT_PATH)
                except Exception as e: # re-try in legacy mode
                    print(e)
                    model = make_model(True)
                    model.load_weights(LOAD_WEIGHT_PATH)
            print(f"Weights loaded : {LOAD_WEIGHT_PATH}", flush=True)
        return model


    def configure_metrics_iterator(iterator):
        def fun(it):
            it.output_central_only=True
            it.incomplete_last_batch_mode = 0
            it.return_label_rank = True
            it.disable_random_transforms(True, True)
        set_to_iterator(iterator, fun)


    def metrics_fun(center_scale, frame_window, category_number:int=0, long_range:bool=True, tracking:bool=True):
        metrics_fun_ = get_metrics_fun(center_scale=center_scale, category=category_number>1, tracking=tracking)
        if tracking:
            def fun(y_true, y_pred):
                fw = frame_window
                n_frame_pairs = fw * 2
                if long_range:
                    n_frame_pairs += (fw - 1) * 2
                d_indices = [fw - 1, n_frame_pairs + fw] # BW & FW (verified)
                lm_indices = [d_indices[0] * 3 + i for i in range(3)] + [d_indices[1] * 3 + i for i in range(3)] # BW & FW (link multiplicity: 3 categories each: single, multiple, null)
                return metrics_fun_(y_pred[0][..., fw:fw + 1], y_pred[1][..., fw:fw + 1], y_pred[5][..., fw*category_number:(fw+1)*category_number] if category_number>1 else None,
                                    tf.gather(y_pred[2], indices=d_indices, axis=-1),
                                    tf.gather(y_pred[3], indices=d_indices, axis=-1),
                                    tf.gather(y_pred[4], indices=lm_indices, axis=-1), y_true[0], y_true[5] if category_number>1 else None, y_true[2], y_true[3],
                                    y_true[4], y_true[-3], y_true[-2], y_true[-1])
        else:
            def fun(y_true, y_pred):
                fw = frame_window
                return metrics_fun_(y_pred[0][..., fw:fw + 1], y_pred[1][..., fw:fw + 1], y_pred[2][..., fw*category_number:(fw+1)*category_number] if category_number>1 else None,
                                    y_true[0], y_true[2] if category_number>1 else None, y_true[-2], y_true[-1])
        return fun


    if args.export_only:
        print(f"export only: init model with weights: {WEIGHT_PATH} (exist: {os.path.exists(WEIGHT_PATH)})", flush=True)
        model = init_model(False)
        # export model
        model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)
        print("model saved", flush=True)
    else:
        print(f"init iterator...", flush=True)
        test_param = config.get("test_data_augmentation_parameters", {})
        if RUN_TEST and "frame_subsampling" in test_param:
            for ds_params in config["dataset_list"]:
                ds_params["data_augmentation"]["frame_subsampling"] = test_param["frame_subsampling"]
        #it_steps = STEP_NUMBER if WORKERS==1 else 0

        test_it = None
        if RUN_TEST:
            train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE)
            test_param = config.get("test_data_augmentation_parameters", {})
            root_path = "/dataTemp" if os.path.exists("/dataTemp") else "/data"
            file_path = os.path.join(root_path, "test_data_augmentation.h5")
            idx = test_param.get("batch_index", -1)
            if idx < 0 or idx >= len(train_it):
                idx = random.randint(0, len(train_it)-1)
            inputs = []
            outputs = []
            if args.test_data_augmentation:
                input_only = test_param.get("input_only", True)
                n_iterations = test_param.get("iteration_number", 10)
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
            else: # test predict
                model = init_model(False)
                model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON))
                input, _ = train_it[idx]
                output = model.predict(input)
                inputs.append(input)
                outputs.append(output)

            train_it.close()
            transpose_axis = [4, 0, 1, 2, 3]
            cnames, lnames = get_input_channel_and_label(config, True)
            if len(cnames) + len(lnames) > 1:
                inputs = transpose_list(inputs)  # (n_it, n_in) -> (n_in, n_it)
                inputs = [np.transpose(np.stack(i, 0), transpose_axis) for i in inputs]
                input_names = cnames + [f"{ln}_{'EDM' if i == 0 else 'CDM'}" for ln in lnames for i in range(2)]
            else:
                inputs = np.stack(inputs, 0)
                inputs = [np.transpose(inputs, transpose_axis)]
                input_names = cnames
            if len(outputs)>0:
                outputs = transpose_list(outputs) # (n_it, n out) -> (n_out, n_it)
                outputs = [np.transpose(np.stack(o, 0), transpose_axis) for o in outputs]
                output_name = ["EDM", "CDM", "dY", "dX", "LinkMultiplicity", "Category"]
            print(f"writing {len(outputs)+len(inputs)} x {inputs[0].shape} to file: {file_path}", flush=True)
            with h5py.File(file_path, mode='w') as h5pyFile :
                for i, o in enumerate(inputs):
                    h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/input_{i}_{input_names[i]}", data=o)
                if len(outputs)>0:
                    for i, o in enumerate(outputs):
                        h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/output_{i}_{output_name[i]}", data=o)

        elif args.compute_metrics:
            model = init_model(False)
            model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON))
            predict_fun = lambda x: model(x, training=False)
            hsm_it = get_iterator(config, init_iterator, step_number=0, shuffle=False, hsm=True)
            configure_metrics_iterator(hsm_it)
            hard_sample_mining_param = t_p.get("hard_sample_mining", {})
            center_scale = hard_sample_mining_param.get("center_scale", 4) if hard_sample_mining_param is not None else 4
            seg_args = config.get("segmentation", {})
            tracking = not seg_args.get("segment_only", False)
            category_number = config["model_architecture"].get("category_number", 0)
            metrics, (batch_size, n_tiles) = compute_metrics(hsm_it, predict_fun, metrics_fun(center_scale=center_scale, frame_window=config["model_architecture"].get("frame_window", 3), category_number = category_number, tracking=tracking), disable_augmentation=True, disable_channel_postprocessing=True, verbose=2)
            if isinstance(batch_size, (list, tuple)):
                tile_column = np.concatenate([np.tile(np.arange(n_t), b_s) for b_s, n_t in zip(batch_size, n_tiles)], axis=0)
            else:
                tile_column = np.tile(np.arange(n_tiles), batch_size)
            header = "IoU;CenterPosition;CenterValue"
            if category_number > 1:
                header +=";Category"
            if tracking:
                header += ";DisplacementL2;LinkMultiplicity"
            if np.any(tile_column != 0):
                header += ";Tile"
                tile_column = tile_column[..., np.newaxis]
                metrics = np.concatenate([metrics, tile_column.astype(metrics.dtype)], axis=1)
            root_path = "/dataTemp" if os.path.exists("/dataTemp") else "/data"
            path = os.path.join(root_path, "metrics.csv")
            print(f"saving metrics of shape: {metrics.shape} to path: {path}", flush=True)
            #print(f"shm n files: {get_shm_nfiles()}")
            np.savetxt(path, metrics, delimiter=";", header=header)
        else: # training
            train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE)
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
            print("init model...", flush=True)

            with strategy.scope():
                model = init_model(training=True)
                model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON))
            
            # perform training
            checkpoint = SafeModelCheckpoint(WEIGHT_PATH, monitor='val_loss' if test_it is not None else 'loss', verbose=1, save_best_only=False, save_weights_only=True)
            lr_schedule = ReduceLROnPlateau2(min_lr=MIN_LR, factor=0.5, patience=PATIENCE, verbose=1, min_delta=0.001, monitor='val_loss' if test_it is not None else 'loss')
            tensorboard_callback = None #tf.keras.callbacks.TensorBoard(LOG_PATH)
            callbacks = [lr_schedule, checkpoint, tf.keras.callbacks.TerminateOnNaN(), StopOnLR(MIN_LR)]
            if tensorboard_callback is not None:
                callbacks.append(tensorboard_callback)
            log_cb = LogsCallback(LOG_PATH + ".csv", start_epoch=START_EPOCH)
            callbacks.append(log_cb)
            hard_sample_mining_param = t_p.get("hard_sample_mining", None)
            if hard_sample_mining_param is not None:
                predict_fun = lambda x: model(x, training=False)
                period = hard_sample_mining_param.get("period", 0.1)
                if period < 1:
                    period = int(N_EPOCHS * period)
                center_scale = hard_sample_mining_param.get("center_scale", 4)
                start_from = hard_sample_mining_param.get("start_from_epoch", 0)
                print("init hsm iterator...", flush=True)
                hsm_it = get_iterator(config, init_iterator, existing_iterator=train_it, step_number=0, shuffle=False, hsm=True) # needs to be a different iterator as iterator.return_central_only
                configure_metrics_iterator(hsm_it)
                seg_args = config.get("segmentation", {})
                tracking = not seg_args.get("segment_only", False)
                arch_params = config["model_architecture"]
                hsm_cb = HardSampleMiningCallback(hsm_it, train_it, predict_fun, metrics_fun(center_scale=center_scale, frame_window=config["model_architecture"].get("frame_window", 3), category_number = arch_params.get("category_number", 0), tracking=tracking), period, start_epoch=START_EPOCH, start_from_epoch=start_from, enrich_factor=hard_sample_mining_param.get("enrich_factor", 100), quantile_max=hard_sample_mining_param.get("quantile_max", None), quantile_min=hard_sample_mining_param.get("quantile_min", None), verbose=2)
                callbacks.append(hsm_cb)

            else:
                hsm_it = None
                hsm_cb = None

            N_EPOCHS -= START_EPOCH
            if N_EPOCHS > 0:
                if WORKERS > 1:
                    # check available shm:
                    shm = get_shm_info()
                    if shm is not None and shm[2] < 1:
                        print( f"Warning: available shared memory is low: {shm[2]:.2f}/{shm[0]:.2f}G, this can hamper multiprocessing", force=True)
                    #enq = tf.keras.utils.OrderedEnqueuer(train_it, use_multiprocessing=True, shuffle=True)
                    enq = OrderedEnqueuerCF(train_it, shuffle=True)
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
            elif START_EPOCH > 0:
                print("Start Epoch is greater than Epoch number.", flush=True)
            train_it.close(force=True)
            if hsm_cb is not None:
                hsm_cb.close()

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
                    model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)
                    print("model saved", flush=True)

