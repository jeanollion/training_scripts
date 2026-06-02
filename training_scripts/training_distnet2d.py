import argparse
import math
import os, sys, json
import shutil
import platform
import random
import time
import h5py
import copy
import numpy as np
import tensorflow as tf
from tensorflow.keras import mixed_precision
from importlib.metadata import version

from tensorflow.keras.optimizers.schedules import CosineDecay

from dataset_iterator.validation_callback import ValidationCallback
from dataset_iterator.image_data_generator import get_image_data_generator, data_generator_to_channel_postprocessing_fun
from dataset_iterator.datasetIO import MemoryIO, get_datasetIO
from dataset_iterator import extract_tile_random_zoom_function, ConcatIterator
from dataset_iterator.utils import transpose_list
from dataset_iterator.hard_sample_mining import HardSampleMiningCallback, compute_metrics
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF
from dataset_iterator.keras_callbacks import StopOnLR, LogsCallback, SafeModelCheckpoint, ReduceLROnPlateau2, \
    LogLRCallback
from distnet_2d.data import DistnetIterator
from distnet_2d.data.distnet_iterator import ARRAY_KEYWORDS
from distnet_2d.data.swim1d import get_swim1d_function
from distnet_2d.model.architectures import get_architecture
from distnet_2d.model.distnet_2d import get_distnet_2d
from distnet_2d.utils.callbacks import ClassWeightScheduler, GradientMonitorCallback, EpsilonCosineDecayCallback, \
    CosineDecayResume, LogGradientCallback
from distnet_2d.utils.helpers import get_background_foreground_counts, count_links
from distnet_2d.utils.metrics_tf import get_metrics_fun
from training_core import open_config_file, get_iterator, chain_pp_fun, set_to_iterator, should_load_dataset_in_shm, \
    get_shm_info, get_input_channel_and_label, get_category_class_counts, compute_category_weights, check_requirements, \
    print_requirement_error, compare_versions, reinitialize_weights, export_fp16_model, \
    compute_category_keep_probabilities

DETERMINISTIC_MODE = True # for testing purposes
if DETERMINISTIC_MODE:
    import random
    SEED = int(os.environ.get("TRAIN_SEED", "42"))
    os.environ["PYTHONHASHSEED"] = str(SEED)
    random.seed(SEED)
    np.random.seed(SEED)
    tf.random.set_seed(SEED)
    tf.keras.utils.set_random_seed(SEED)
    random.seed(42)

__VERSION__ = '1.1.6'
__REQUIRES__ = ["dataset_iterator>=0.5.8", "distnet2d>=0.2.5" ]

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
parser.add_argument("--min_script_version", type=str, help="minimal script version")
parser.add_argument("--mixed_precision", action="store_true", help="Mixed Precision (float16) training")
parser.add_argument("--export_fp16", action="store_true", help="Export to float16 Precision")

if __name__ == "__main__":
    args = parser.parse_args()
    # check script version and requirements:
    if not check_requirements(__REQUIRES__):
        sys.exit(1)
    if args.min_script_version:
        if compare_versions(__VERSION__, args.min_script_version) < 0:
            print(f"script version is out-of-date: {__VERSION__} minimal version: {args.min_script_version}", flush=True)
            print_requirement_error()
            sys.exit(1)

    if DETERMINISTIC_MODE: # for debugging purposes
        import subprocess
        try:
            h = subprocess.check_output(["cat", "HEAD"], cwd="/distnet2d/.git", stderr=subprocess.DEVNULL).decode().strip()
            print(f"distnet2d commit: {h}")
        except Exception as e:
            print(f"distnet2d commit: unknown ({e})")

    print(f"Script version: {__VERSION__}; dataset_iterator version: {version('dataset_iterator')}; DiSTNet2D version: {version('DiSTNet2D')}  tensorflow : {tf.__version__} python: {platform.python_version()}")
    if args.mixed_precision:
        mixed_precision.set_global_policy('mixed_float16')
        print(f"Mixed precision policy= {mixed_precision.global_policy()}")
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
    WARMUP_EPOCHS = max(2, int(N_EPOCHS / 50))
    STEP_NUMBER = args.step_number if args.step_number is not None else t_p.get("step_number", 200)
    VAL_STEP_NUMBER = t_p.get("validation_step_number", 100)
    VAL_FREQ = t_p.get("validation_frequency", 1)
    PATIENCE = args.patience if args.patience is not None else t_p.get("patience", 80)
    LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
    MIN_LR = args.min_learning_rate if args.min_learning_rate is not None else t_p.get("min_learning_rate", 5e-7)
    EPSILON = 1e-7 # trade-off: higher -> learns slower but stabilises (especially with FP16). keras default is 1e-7
    MIN_EPSILON = 1e-7
    if args.strategy == "multiworker-slurm":
        WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    else:
        WORKERS = t_p.get("multiprocessing_workers", 1)
    WORKERS = min(os.cpu_count(), WORKERS)
    SHUFFLE = not RUN_TEST
    START_EPOCH = t_p.get("start_epoch", 0)
    print(f"configuration file found. ")

    def init_iterator(ds_conf, step_number, dataset=None, category_class_counts:list[int] = None, **kwargs):
        data_aug_params = ds_conf.get("data_augmentation", {})
        seg_args = config.get("segmentation", {})
        tracking = seg_args.get("tracking", not seg_args.get("segment_only", False))
        segmentation = seg_args.get("segmentation", True)
        print(f"segmentation: {segmentation} tracking: {tracking}")
        arch_params = config["model_architecture"]
        category_number = arch_params.get("category_number", 0)
        channel_names = ds_conf.get("channel_name", "raw")
        default_frame_aware = config["model_architecture"]["architecture_type"].lower() == "tempy" # TODO change this mechanism
        frame_aware = arch_params.get("frame_aware", default_frame_aware)
        if arch_params["frame_window"] == 0:
            frame_aware = False
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
            sp["group_keyword"] = ds_conf.get("keyword", None)
        affine_transform_parameters = data_aug_params.get("affine_transform_parameters", None)
        data_generators = [get_image_data_generator(scaling_parameters=sp, affine_transform_parameters=affine_transform_parameters) for sp in scaling_parameters]
        affine_transform_parameters_mask = None if affine_transform_parameters is None else {**affine_transform_parameters, "interpolation_order": 0}
        mask_generator = get_image_data_generator(scaling_parameters=[], affine_transform_parameters=affine_transform_parameters_mask)
        pp_fun_list = []
        swim1D_params = data_aug_params.get("swim1d_parameters", None)
        if swim1D_params is not None:
            mask_channels = [1] + [cidx + len(channel_names) + 1 for cidx in range(0, len(label_names))]
            pp_fun_list.append(get_swim1d_function(mask_channels,
                                                   ref_mask_idx=swim1D_params.get("ref_mask_idx", 0),
                                                   distance=swim1D_params.get("distance", 50),
                                                   min_gap=swim1D_params.get("min_gap", 3),
                                                   closed_end=swim1D_params.get("closed_end", True)))
        # perform illumination at the end: after elastic deform and swim
        illumination_parameters = data_aug_params.get("illumination_transform", [data_aug_params.get("illumination_parameters", None)])
        for cidx, ip in enumerate(illumination_parameters):
            if ip is not None:
                pp_fun_list.append(data_generator_to_channel_postprocessing_fun(get_image_data_generator(illumination_parameters=ip), [0 if cidx ==0 else cidx + 1])) # channel #1 is reserved to labels

        pp_fun = chain_pp_fun(pp_fun_list)
        fw = arch_params["frame_window"]
        if fw == 0:
            frame_aware = False
        iterator_params = dict(erase_edge_cell_size=data_aug_params.get("erase_edge_cell_size", 0),
                               aug_remove_prob=data_aug_params.get("static_probability", 0.01),
                               future_frames=arch_params.get("next", True),
                               scale_edm = seg_args.get("scale_edm", False),
                               center_mode=seg_args.get("center_mode", "MEDOID"),
                               center_distance_mode=seg_args.get("center_distance_mode", "GEODESIC"),
                               frame_window=fw,
                               image_data_generators=[data_generators[0], mask_generator] + data_generators[1:],
                               elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None),
                               void_mask_proportion = [0, 0] if fw == 0 else None,  # exclude empty frame, only when no frame window
                               channels_postprocessing_function=pp_fun, verbose=False and RUN_TEST, memory_persistent=memory_persistent)
        dataset_parameters = config.get("dataset_parameters", {})
        tridimensional_mode = len(dataset_parameters.get("input_shape", [None, None])) == 3
        z_radius = dataset_parameters.get("z_radius", 1.)
        if category_number > 1 and category_class_counts is not None:
            cat_loss_param = seg_args.get("category_loss_parameters", {})
            return_loss_mask = cat_loss_param.get("class_balanced_loss_masking", False)
            if return_loss_mask:
                category_keep_probabilities = compute_category_keep_probabilities( category_class_counts, power_law=cat_loss_param.get("weight_power_law", 1)) if return_loss_mask else None
            else:
                category_keep_probabilities = None
        else:
            return_loss_mask = False
            category_keep_probabilities = None
        array_kw = (ARRAY_KEYWORDS[:1] if tracking else []) + (ARRAY_KEYWORDS[1:] if category_number>1 else [])
        return DistnetIterator(dataset=dataset, channel_keywords=[channel_names[0], '/regionLabels'] + channel_names[1:],
                               input_label_keywords=label_names, array_keywords=array_kw,
                               input_label_center_idx = seg_args.get("input_label_center_idx", -1),
                               segmentation=segmentation,
                               tracking = tracking,
                               frame_aware = frame_aware,
                               tridimensional_mode=tridimensional_mode,
                               z_radius=z_radius,
                               return_loss_mask=return_loss_mask,
                               category_keep_probabilities=category_keep_probabilities,
                               group_keyword=ds_conf.get("keyword", None),
                               batch_size=batch_size, step_number=step_number, extract_tile_function=extract_tiles_fun,
                               return_edm_derivatives=seg_args.get("edm_derivatives", False), return_cdm_derivatives=False, # no need to return cdm derivatives: computed at train time
                               aug_frame_subsampling=data_aug_params.get("frame_subsampling", 1), shuffle=kwargs.get("shuffle", True),
                               **iterator_params)


    def get_edm_class_weights(config: dict, power_law:float = 1, max_weight:float=None):
        counts = np.array([0, 0], dtype="float128")
        for i, ds_conf in enumerate(config["dataset_list"]):
            counts += get_background_foreground_counts(ds_conf["path"], channel_keyword='/regionLabels', group_keyword=ds_conf.get("keyword", None))
        weights = compute_category_weights(counts.tolist(), power_law=power_law, max_weight=max_weight)
        return weights.astype("float32")

    def get_link_multiplicity_class_weights(config: dict, max_weight= 50, power_law:float=1):
        counts = [0] * 3
        for i, ds_conf in enumerate(config["dataset_list"]):
            dataset = get_datasetIO(ds_conf["path"], 'r')
            paths = dataset.get_dataset_paths(ARRAY_KEYWORDS[0], ds_conf.get("keyword", None))
            for p in paths:
                lm_array = dataset.get_dataset(p)
                s, m, n = count_links(lm_array, detailed = False)
                counts[0] += s
                counts[1] += m
                counts[2] += n
            dataset.close()
        print(f"link multiplicity counts: {[int(x) for x in counts]}")
        return compute_category_weights(counts, max_weight=max_weight, power_law = power_law)


    def init_model(training:bool, category_class_counts:list[int]=None):
        arch_args = copy.deepcopy(config["model_architecture"])
        seg_args = config.get("segmentation", {})
        tracking = seg_args.get("tracking", not seg_args.get("segment_only", False))
        segmentation = seg_args.get("segmentation", True)
        shape = config["dataset_parameters"]["input_shape"]
        input_shape = [None if s <= 0 else s for s in shape]
        nchan, nlabel = get_input_channel_and_label(config)
        n_inputs = nchan + nlabel * 2 # for each label EDM and GDCM are added
        tracking_args = config.get("tracking", {})
        lm_loss_params = tracking_args.get("lm_loss_parameters", {})
        if training and tracking and lm_loss_params.get("weight_power_law", 1)>0:
            link_multiplicity_class_weights = get_link_multiplicity_class_weights(config,  max_weight=lm_loss_params.get("max_weight", 50), power_law = lm_loss_params.get("weight_power_law", 1))
            print(f"link multiplicity weights: { {l:float(w) for l,w in zip(['single', 'multiple', 'null'], link_multiplicity_class_weights)} }")
        else:
            link_multiplicity_class_weights = [1, 1, 1]
        lm_focal_weight = lm_loss_params.get("focal_weight", 1)
        category_number = arch_args.get("category_number", 0)
        cat_loss_params = seg_args.get("category_loss_parameters", {})
        if training and category_number > 1 and cat_loss_params.get("weight_power_law", 1)>0 and category_class_counts is not None:
            category_class_weights = compute_category_weights(category_class_counts, power_law=cat_loss_params.get("weight_power_law", 1), max_weight=cat_loss_params.get("max_weight", 0))
        else:
            category_class_weights = [1] * category_number
        return_loss_mask = cat_loss_params.get("class_balanced_loss_masking", False)
        balance_edm_weight = "balance_edm_frequency_parameters" in seg_args
        edm_class_weights = get_edm_class_weights(config, power_law = seg_args["balance_edm_frequency_parameters"].get("weight_power_law", 1)) if training and balance_edm_weight else None
        if edm_class_weights is not None:
            print(f"edm background/foreground balancing weights {edm_class_weights}")
        arch = get_architecture(arch_args.pop("architecture_type", "blend"),
                                scale_edm=seg_args.get("scale_edm", False),
                                spatial_dimensions=input_shape, n_inputs=n_inputs, segmentation=segmentation, tracking=tracking, **arch_args)
        cdm_loss_radius = seg_args.get("cdm_loss_radius", 0)
        if training: # perform test step if at least one test dataset
            ds_types = [ ds_conf.get("type", "TRAIN") for ds_conf in config["dataset_list"] ]
            perform_test_step = "TEST" in ds_types
        else:
            perform_test_step = False

        def make_model(legacy:bool=False):
            return get_distnet_2d(arch=arch,
                                  gradient_accumulation_steps = config["training_parameters"].get("gradient_accumulation_steps", 1),
                                  edm_class_weights=edm_class_weights,
                                  edm_derivative_loss=seg_args.get("edm_derivatives", True),
                                  cdm_derivative_loss=seg_args.get("cdm_derivatives", True), cdm_loss_radius=cdm_loss_radius,
                                  link_multiplicity_class_weights=link_multiplicity_class_weights, link_multiplicity_focal_weight=lm_focal_weight,
                                  link_multiplicity_temperature=lm_loss_params.get("temperature", 0), link_multiplicity_label_smoothing=lm_loss_params.get("label_smoothing", 0),
                                  category_class_weights=category_class_weights, category_focal_weight = cat_loss_params.get("focal_weight", 2),
                                  category_temperature=cat_loss_params.get("temperature", 0), category_label_smoothing=cat_loss_params.get("label_smoothing", 0),
                                  perform_test_step=perform_test_step, scale_losses = not legacy,
                                  return_weight_map=return_loss_mask)
        model = make_model()
        if args.export_only or ( (args.compute_metrics or args.test_predict) and os.path.exists(WEIGHT_PATH)):
            assert os.path.exists(WEIGHT_PATH), f"weights {WEIGHT_PATH} not found"
            try:
                model.load_weights(WEIGHT_PATH) # , by_name=True
            except Exception as e: # re-try in legacy mode
                print(e)
                model = make_model(True)
                model.load_weights(WEIGHT_PATH) # , by_name=True

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
            #print(f"model loss scales: {model.loss_scales}")

        #from training_core import dump_layer_config
        #with open(os.path.join(args.config_dir, "model_config.json"), "w") as f:
        #    json.dump(dump_layer_config(model), f, indent=2, default=str)

        return model


    def configure_metrics_iterator(iterator):
        def fun(it):
            it.output_central_only=True
            it.incomplete_last_batch_mode = 0
            it.return_label_rank = True
            it.disable_random_transforms(True, True)
        set_to_iterator(iterator, fun)


    def metrics_fun(scale, frame_window, category_number:int=0, long_range:bool=True, segmentation:bool=True, tracking:bool=True, tridimensional_mode:bool=False, exclusion_weight_map:bool=False):
        metrics_fun_ = get_metrics_fun(scale, category=category_number>1, segmentation=segmentation, tracking=tracking, tridimensional_mode=tridimensional_mode)
        _wm_offset = 1 if exclusion_weight_map else 0
        if tracking:
            assert segmentation
            # output indices: [EDM, CDM, (dZ), dY, dX, LM, Cat, label, prev label, center object, (WeightMap)]
            dz_idx = 2
            dy_idx = 3 if tridimensional_mode else 2
            dx_idx = 4 if tridimensional_mode else 3
            lm_idx = 5 if tridimensional_mode else 4
            cat_idx = 6 if tridimensional_mode else 5
            label_idx = -3 - _wm_offset
            prev_label_idx = -2 - _wm_offset
            center_idx = - 1 - _wm_offset
            def fun(y_true, y_pred):
                fw = frame_window
                n_frame_pairs = fw * 2
                if long_range:
                    n_frame_pairs += (fw - 1) * 2
                d_indices = [fw - 1, n_frame_pairs + fw] # BW & FW (verified)
                lm_indices = [d_indices[0] * 3 + i for i in range(3)] + [d_indices[1] * 3 + i for i in range(3)] # BW & FW (link multiplicity: 3 categories each: single, multiple, null)
                return metrics_fun_(y_pred[0][..., fw:fw + 1], y_pred[1][..., fw:fw + 1], y_pred[cat_idx][..., fw*category_number:(fw+1)*category_number] if category_number>1 else None,
                                    tf.gather(y_pred[dz_idx], indices=d_indices, axis=-1) if tridimensional_mode else None,
                                    tf.gather(y_pred[dy_idx], indices=d_indices, axis=-1),
                                    tf.gather(y_pred[dx_idx], indices=d_indices, axis=-1),
                                    tf.gather(y_pred[lm_idx], indices=lm_indices, axis=-1),
                                    y_true[0][..., :1], # in case iterator returns edm derivatives
                                    y_true[cat_idx] if category_number>1 else None,
                                    y_true[dz_idx] if tridimensional_mode else None, y_true[dy_idx], y_true[dx_idx],
                                    y_true[lm_idx],
                                    y_true[label_idx], y_true[prev_label_idx], y_true[center_idx]) # label, prev_label, centerobject
        elif segmentation:
            # output indices: [EDM, CDM, Cat, label, center object, (WeightMap)]
            label_idx = -2 - _wm_offset
            center_idx = - 1 - _wm_offset
            def fun(y_true, y_pred):
                fw = frame_window
                return metrics_fun_(y_pred[0][..., fw:fw + 1], y_pred[1][..., fw:fw + 1], y_pred[2][..., fw*category_number:(fw+1)*category_number] if category_number>1 else None,
                                    y_true[0][..., :1], # in case iterator returns edm derivatives
                                    y_true[2] if category_number>1 else None,
                                    y_true[label_idx], y_true[center_idx]) # label, center_object
        else: # category only
            assert category_number > 1, f"category_number {category_number} must be > 1"
            label_idx = -2 - _wm_offset
            center_idx = - 1 - _wm_offset
            def fun(y_true, y_pred):
                fw = frame_window
                return metrics_fun_(y_pred[0][..., fw*category_number:(fw+1)*category_number], y_true[1], y_true[label_idx], y_true[center_idx])
        return fun


    def init_loss_scales(model, train_it, steps:int=30, reinit_independent:bool=False):
        if len(model.get_sub_losses_names()) == 1:
            return
        steps = min(len(train_it), steps)
        losses_names = model.get_sub_losses_names()
        acc_losses = {k: [] for k in losses_names}
        if steps>0:
            print(f"initializing loss scales (reinit independent={reinit_independent})...", flush=True)
            @tf.function
            def run_batch(model, data):
                return model.test_step(data)

            enq = OrderedEnqueuerCF(train_it, shuffle=True, name="init_test", max_steps=steps)
            enq.start(workers=WORKERS, max_queue_size=max(1, min(STEP_NUMBER - 1,  WORKERS)))
            gen = enq.get()
            for i in range(steps):
                data = next(gen)
                print(f"{i+1}/{steps}", flush=True)
                losses = run_batch(model, data)
                if i%3==0:
                    reinitialize_weights(model, independent=reinit_independent)
                for k in losses_names:
                    acc_losses[k].append(losses[k])
            enq.stop()
            del enq
            mean_losses = [np.mean(acc_losses[k]) for k in losses_names]
            print(f"loss scales init values: {[float(m) for m in mean_losses]}", flush=True)
        else:
            # steps == 0: skip loss-scale measurement but still apply the weight reinit.
            # With independent=False (default), every layer's seeded initializer is rebuilt
            # with the same base seed -> all layers draw scaled copies of the same base
            # pattern (inter-layer correlation ~1). Combined with per-layer normalization
            # this acts as an orthogonal/identity-style warm-up and accelerates early
            # training. With independent=True, each weight gets a unique seed and the
            # reinit produces a regular (uncorrelated) random init, reproducible across runs.
            print(f"reinitializing weights (skipping loss scale measurement, independent={reinit_independent})...", flush=True)
            reinitialize_weights(model, independent=reinit_independent)
            mean_losses = [1] * len(losses_names)
        if isinstance(getattr(model, "loss_scales", None), tf.Variable):
            model.loss_scales.assign(mean_losses)

    if args.export_only:
        print(f"export only: init model with weights: {WEIGHT_PATH} (exist: {os.path.exists(WEIGHT_PATH)}) fp16: {args.export_fp16}", flush=True)
        model = init_model(False)
        if args.export_fp16:
            export_fp16_model(model, SAVED_MODEL_PATH)
        else:
            model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)
        print("model saved", flush=True)
    else:
        print(f"init iterator...", flush=True)
        test_param = config.get("test_data_augmentation_parameters", {})
        if RUN_TEST and "frame_subsampling" in test_param:
            for ds_params in config["dataset_list"]:
                ds_params["data_augmentation"]["frame_subsampling"] = test_param["frame_subsampling"]
        #it_steps = STEP_NUMBER if WORKERS==1 else 0

        if RUN_TEST:
            seg_args = config.get("segmentation", {})
            tracking = seg_args.get("tracking", not seg_args.get("segment_only", False))
            segmentation = seg_args.get("segmentation", True)
            category_number = config["model_architecture"].get("category_number", 0)
            category_only = not segmentation and not tracking and category_number > 0
            return_loss_mask = seg_args.get("category_loss_parameters", {}).get("class_balanced_loss_masking", False)
            category_class_counts = get_category_class_counts(config, category_number, category_keyword=ARRAY_KEYWORDS[1]) if category_number > 1 and return_loss_mask else None
            train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE, category_class_counts=category_class_counts)
            test_param = config.get("test_data_augmentation_parameters", {})
            root_path = "/dataTemp" if os.path.exists("/dataTemp") else "/data"
            file_path = os.path.join(root_path, "test_data_augmentation.h5")
            arch_params = config["model_architecture"]
            default_frame_aware = arch_params["architecture_type"].lower() == "tempy" # TODO change this mechanism
            frame_aware = arch_params.get("frame_aware", default_frame_aware)
            if arch_params["frame_window"] == 0:
                frame_aware = False
            tridimensional_mode = len(config.get("dataset_parameters", {}).get("input_shape", [None, None])) == 3
            if segmentation and tracking:
                if tridimensional_mode:
                    output_name = ["EDM", "CDM", "dZ", "dY", "dX", "LinkMultiplicity", "Category"]
                else:
                    output_name = ["EDM", "CDM", "dY", "dX", "LinkMultiplicity", "Category"]
            elif segmentation:
                output_name = ["EDM", "CDM", "Category"]
            elif category_only:
                output_name = ["Category"]
            else:
                raise ValueError("Invalid configuration: allowed mode = segmentation + tracking (+category) / segmentation (+category) / category only")
            if category_number > 1 and return_loss_mask:
                output_name.append("LossMask")
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
                    if frame_aware:
                        input = input[:-1]
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
                model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON)) # test : , clipnorm=1.0
                input, _ = train_it[idx]
                output = model.predict(input)
                if frame_aware:
                    input = input[:-1]
                inputs.append(input)
                outputs.append(output)

            train_it.close()
            transpose_axis = [5, 0, 1, 2, 3, 4] if tridimensional_mode else [4, 0, 1, 2, 3]
            cnames, lnames = get_input_channel_and_label(config, True)

            inputs = transpose_list(inputs)  # (n_it, n_in) -> (n_in, n_it)
            inputs = [np.transpose(np.stack(i, 0), transpose_axis) for i in inputs]
            input_names = cnames + [f"{ln}_{'EDM' if i == 0 else 'CDM'}" for ln in lnames for i in range(2)]

            if len(outputs)>0:
                if len(output_name) > 1 or isinstance(outputs[0], (list, tuple)):
                    outputs = transpose_list(outputs) # (n_it, n out) -> (n_out, n_it)
                    outputs = [np.transpose(np.stack(o, 0), transpose_axis) for o in outputs]
                else:
                    outputs = [np.transpose(np.stack(outputs, 0), transpose_axis)]
            if tridimensional_mode:
                if config["model_architecture"].get("frame_window", 3)>0: # remove batch axis
                    inputs = [a[:, :, 0] for a in inputs]
                    outputs = [a[:, :, 0] for a in outputs]
                else: # remove time axis
                    inputs = [a[0] for a in inputs]
                    outputs = [a[0] for a in outputs]

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
            hsm_it = get_iterator(config, init_iterator, step_number=0, shuffle=False, hsm=True) # do not return loss mask
            configure_metrics_iterator(hsm_it)
            hard_sample_mining_param = t_p.get("hard_sample_mining", {})
            scale = hard_sample_mining_param.get("scale", hard_sample_mining_param.get("center_scale", 4) * 2) if hard_sample_mining_param is not None else 8
            seg_args = config.get("segmentation", {})
            tracking = seg_args.get("tracking", not seg_args.get("segment_only", False))
            segmentation = seg_args.get("segmentation", True)
            category_number = config["model_architecture"].get("category_number", 0)
            tridimensional_mode = len(config.get("dataset_parameters", {}).get("input_shape", [None, None])) == 3
            category_only = not segmentation and not tracking and category_number > 0
            metrics, (batch_size, n_tiles) = compute_metrics(hsm_it, predict_fun, metrics_fun(scale=scale, frame_window=config["model_architecture"].get("frame_window", 3), category_number = category_number, segmentation=segmentation, tracking=tracking, tridimensional_mode=tridimensional_mode, exclusion_weight_map=False), disable_augmentation=True, disable_channel_postprocessing=True, verbose=2)
            if isinstance(batch_size, (list, tuple)):
                tile_column = np.concatenate([np.tile(np.arange(n_t), b_s) for b_s, n_t in zip(batch_size, n_tiles)], axis=0)
            else:
                tile_column = np.tile(np.arange(n_tiles), batch_size)
            header = "IoU;FPD;CenterPosition;CenterValue" #
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
            category_number = config["model_architecture"].get("category_number", 0)
            category_class_counts = get_category_class_counts(config, category_number, category_keyword=ARRAY_KEYWORDS[1]) if category_number > 1 else None
            train_it = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE, category_class_counts = category_class_counts)
            val_it = get_iterator(config, init_iterator, step_number=VAL_STEP_NUMBER, shuffle=SHUFFLE, dataset_type="TEST", category_class_counts = category_class_counts)
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
                model = init_model(training=True, category_class_counts=category_class_counts)
                learning_rate = CosineDecayResume(initial_learning_rate=LR,
                                            min_lr=MIN_LR,
                                            decay_steps=STEP_NUMBER * N_EPOCHS,
                                            start_step = STEP_NUMBER * START_EPOCH,
                                            warmup_learning_rate_factor=1./10,
                                            warmup_steps = STEP_NUMBER * WARMUP_EPOCHS)
                accum_steps = config["training_parameters"].get("gradient_accumulation_steps", 1)
                beta_1 = 0.9 ** (1.0 / accum_steps)
                beta_2 = 0.999 ** (1.0 / accum_steps)
                model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate, beta_1=beta_1, beta_2=beta_2, epsilon=EPSILON))

                if LOAD_WEIGHT_PATH is None:
                    # steps>0 : measure per-head loss means AND apply reinit during the loop.
                    # steps==0: skip measurement, only apply reinit.
                    # steps<0 : skip entirely (no reinit, no scales).
                    # reinit_independent: False (default) -> correlated scaled-copies init
                    # (faster early training);  True -> uncorrelated regular random init.
                    init_loss_scales_steps = config["training_parameters"].get("init_loss_scales_steps", 30)
                    reinit_independent = config["training_parameters"].get("reinit_independent", False)
                    if init_loss_scales_steps >= 0:
                        init_loss_scales(model, train_it, steps=init_loss_scales_steps, reinit_independent=reinit_independent)

            # perform training
            checkpoint = SafeModelCheckpoint(WEIGHT_PATH, monitor='val_loss' if val_it is not None and VAL_FREQ==1 else 'loss', verbose=1, save_best_only=False, save_weights_only=True)
            tensorboard_callback = None #tf.keras.callbacks.TensorBoard(LOG_PATH)
            callbacks = [LogLRCallback(), checkpoint, tf.keras.callbacks.TerminateOnNaN(), StopOnLR(MIN_LR)]
            callbacks.append(LogGradientCallback(STEP_NUMBER))
            if MIN_EPSILON<EPSILON:
                eps_cb = EpsilonCosineDecayCallback(start_epsilon=EPSILON, min_epsilon=MIN_EPSILON, decay_steps=STEP_NUMBER * WARMUP_EPOCHS)
                callbacks.append(eps_cb)
            if tensorboard_callback is not None:
                callbacks.append(tensorboard_callback)
            #callbacks.append(GradientMonitorCallback(STEP_NUMBER))
            log_cb = LogsCallback(LOG_PATH + ".csv", start_epoch=0)
            callbacks.append(log_cb)

            if "balance_edm_frequency_parameters" in config.get("segmentation", {}):
                freq_param = config["segmentation"]["balance_edm_frequency_parameters"]
                if freq_param.get("dynamic_weights", True):
                    callbacks.append(ClassWeightScheduler(attribute_name = "edm_class_weights", n_epochs=N_EPOCHS, power_law = freq_param.get("dynamic_power_law", 1)))
            category_number = config["model_architecture"].get("category_number", 0)

            hard_sample_mining_param = t_p.get("hard_sample_mining", None)
            if hard_sample_mining_param is not None:
                predict_fun = lambda x: model(x, training=False)
                period = hard_sample_mining_param.get("period", 0.1)
                if period < 1:
                    period = int(N_EPOCHS * period)
                scale = hard_sample_mining_param.get("scale", hard_sample_mining_param.get("center_scale", 4) * 2)
                start_from = hard_sample_mining_param.get("start_from_epoch", 0)
                print("init hsm iterator...", flush=True)
                hsm_it = get_iterator(config, init_iterator, existing_iterator=train_it, step_number=0, shuffle=False, hsm=True) # needs to be a different iterator as iterator.return_central_only. do not return loss mask
                configure_metrics_iterator(hsm_it)
                seg_args = config.get("segmentation", {})
                tracking = seg_args.get("tracking", not seg_args.get("segment_only", False))
                segmentation = seg_args.get("segmentation", True)
                arch_params = config["model_architecture"]
                tridimensional_mode = len(config.get("dataset_parameters", {}).get("input_shape", [None, None])) == 3
                category_number = arch_params.get("category_number", 0)
                hsm_cb = HardSampleMiningCallback(hsm_it, train_it, predict_fun, metrics_fun(scale=scale, frame_window=config["model_architecture"].get("frame_window", 3), category_number = category_number, segmentation=segmentation, tracking=tracking, tridimensional_mode=tridimensional_mode, exclusion_weight_map=False), n_epochs=N_EPOCHS, period=period, start_epoch=START_EPOCH, start_from_epoch=start_from, enrich_factor=hard_sample_mining_param.get("enrich_factor", 100), quantile_max=hard_sample_mining_param.get("quantile_max", None), quantile_min=hard_sample_mining_param.get("quantile_min", None), verbose=2)
                callbacks.append(hsm_cb)
            else:
                hsm_it = None
                hsm_cb = None
            if val_it is not None: # Important: add after HSM cb
                val_cb = ValidationCallback(val_it, STEP_NUMBER, validation_freq=VAL_FREQ, start_epoch=START_EPOCH)
                callbacks.append(val_cb)
            else:
                val_cb = None

            if N_EPOCHS > START_EPOCH:
                if WORKERS > 1:
                    # check available shm:
                    shm = get_shm_info()
                    if shm is not None and shm[2] < 1:
                        print( f"Warning: available shared memory is low: {shm[2]:.2f}/{shm[0]:.2f}G, this can hamper multiprocessing", force=True)
                    enq = OrderedEnqueuerCF(train_it, shuffle=True, name="main")
                    if hsm_cb is not None:
                        hsm_cb.set_enqueuer(enq)

                    if val_it is not None:
                        val_enq = OrderedEnqueuerCF(val_it, shuffle=False, name="val", max_steps=VAL_STEP_NUMBER)
                        val_cb.set_enqueuer(val_enq, enq)
                        val_enq.start(workers=WORKERS, max_queue_size=max(1, min(VAL_STEP_NUMBER - 1,  WORKERS)))
                        val_gen = val_enq.get(block=False, name="val")
                    else:
                        val_gen = None
                    enq.start(workers=WORKERS, max_queue_size=max(1, min(STEP_NUMBER - 1,  WORKERS)))  # queue should be lower than step number otherwise there can be stalling when several enqueueurs are running
                    gen = enq.get()
                else:
                    gen = train_it
                    val_gen = val_it
                if hsm_cb is not None:
                    hsm_cb.initialize()
                if val_cb is not None:
                    val_cb.initialize()

                model.fit(gen, epochs=N_EPOCHS, initial_epoch=START_EPOCH, steps_per_epoch=STEP_NUMBER, validation_data=val_gen, callbacks=callbacks, workers=1, use_multiprocessing=False, validation_steps = VAL_STEP_NUMBER, validation_freq=VAL_FREQ)
                if WORKERS > 1:
                    print("stopping enqueuer...", flush=True)
                    enq.stop()
                    if val_it is not None:
                        val_enq.stop()
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
                    if is_chief:
                        model.load_weights(WEIGHT_PATH)  # reload best weights
                    if args.export_fp16:
                        export_fp16_model(model, save_path)
                    else:
                        model.save(
                            save_path,
                            include_optimizer=False,
                            save_traces=True,
                            inference=True,
                        )
                    print(f"model saved. fp16: {args.export_fp16}", flush=True)

                    if not is_chief:
                        print(f"cleaning temp models at {save_path}", flush=True)
                        shutil.rmtree(save_path)  # clean up for non chief worker
                else:
                    model.load_weights(WEIGHT_PATH)  # reload best weights
                    if args.export_fp16:
                        export_fp16_model(model, SAVED_MODEL_PATH)
                    else:
                        model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True, inference=True)
                    print(f"model saved. fp16: {args.export_fp16}", flush=True)

