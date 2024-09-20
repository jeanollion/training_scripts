import os
import argparse
import random
import numpy as np
import tensorflow as tf
import h5py
import copy
from importlib.metadata import version
from dataset_iterator.datasetIO import MemoryIO
from dataset_iterator import extract_tile_random_zoom_function
from dataset_iterator.utils import transpose_list, is_list
from dataset_iterator.helpers import get_channel_number
from ssnb_denoising.datasets import get_center_scale
from ssnb_denoising.training import train_denoiser, get_train_iterator, get_collapse_test_iterator
from ssnb_denoising.datasets.evaluation import get_eval_iterator
from ssnb_denoising.models import get_dnet, get_dnet_multiframe, get_convolution, BlindDenoiser
from ssnb_denoising.models.dnet_n2n import get_dnet_n2n
from training_core import open_config_file, get_iterator, should_load_dataset_in_shm

__VERSION__ = '1.0.0'
parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--export_only", action="store_true", help="skip model training, only export")
parser.add_argument("--train_only", action="store_true", help="train but no export")
parser.add_argument("--test_data_augmentation", action="store_true", help="generate and store example of augmented data")
parser.add_argument("--compute_metrics", action="store_true", help="compute loss")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
parser.add_argument("--n_epochs", type=int, help="number of training epochs")
parser.add_argument("--step_number", type=int, help="number of training steps per epoch")
parser.add_argument("--learning_rate", type=float, help="initial learning rate for training")
parser.add_argument("--min_learning_rate", type=float, help="minimal learning rate for training")

if __name__ == "__main__":
    args = parser.parse_args()

    # get parameters
    CONFIG = open_config_file(args.config_dir, args.test_data_augmentation)
    t_p = CONFIG["training_parameters"]
    MODEL_NAME = t_p["model_name"] + (f"_{args.model_idx}" if args.model_idx is not None else "")
    WEIGHT_PATH = os.path.join(args.config_dir, t_p["weight_dir"], MODEL_NAME + ".h5") if len(t_p["weight_dir"]) > 0 else os.path.join(args.config_dir, MODEL_NAME + ".h5")
    LOAD_WEIGHT_PATH = t_p["load_model_file"] if len(t_p.get("load_model_file", "")) > 0 else None
    LOG_PATH = os.path.join(args.config_dir, t_p["log_dir"], MODEL_NAME) if len(t_p["log_dir"]) > 0 else os.path.join(args.config_dir, MODEL_NAME)
    SAVED_MODEL_PATH = os.path.join(args.export_dir if args.export_dir is not None else args.config_dir, MODEL_NAME)
    N_EPOCHS = args.n_epochs if args.n_epochs is not None else t_p.get("n_epochs", 800)
    STEP_NUMBER = args.step_number if args.step_number is not None else t_p.get("step_number", 200)
    LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
    MIN_LR = args.min_learning_rate if args.min_learning_rate is not None else t_p.get("min_learning_rate", 1e-6)
    EPSILON_RANGE = t_p.get("epsilon_range", [0.1, 0.2])
    EPSILON_RANGE = [max(EPSILON_RANGE), min(EPSILON_RANGE)]
    WORKERS = min(os.cpu_count(), t_p.get("multiprocessing_workers", 1))
    SHUFFLE = not args.test_data_augmentation
    START_EPOCH = t_p.get("start_epoch", 0)
    DENOISING_PARAMETERS = CONFIG.get("denoising_parameters", {})
    RENOISE_MODE = DENOISING_PARAMETERS["denoising_mode"] == "RENOISE"
    NOISE_CORRELATION_RANGE = DENOISING_PARAMETERS.get("noise_correlation_range", None)
    if NOISE_CORRELATION_RANGE == 0 or (is_list(NOISE_CORRELATION_RANGE) and np.all([n==0 for n in NOISE_CORRELATION_RANGE])):
        NOISE_CORRELATION_RANGE = None
    PSF = DENOISING_PARAMETERS.get("psf", None)

    print(f"Script version: {__VERSION__}; dataset_iterator version: {version('dataset_iterator')}; BliSSeD version: {version('ssnb_denoising')}")
    print(f"configuration file found. ")
    print(f"Deconvolution: {'disabled' if PSF is None else ('kernel' if is_list(PSF) else ('gaussian' if PSF>0 else 'trainable gaussian'))}")
    print(f"Noise Correlation Range: {'No correlation' if NOISE_CORRELATION_RANGE is None else NOISE_CORRELATION_RANGE}")

    def weighted_avg(a_b_count):
        if len(a_b_count) == 0:
            return None
        elif len(a_b_count) == 1:
            return a_b_count[0][0], a_b_count[0][1]
        else:
            a_b_count = np.array(a_b_count)
            a_b_count[:, 2] = a_b_count[:, 2] / np.sum(a_b_count[:, 2])
            a_b_count[:, 0:1] = a_b_count[:, 0:1] * a_b_count[:, 2]
            a_b_count = np.sum(a_b_count, axis=0)
            return a_b_count[0] / a_b_count[2], a_b_count[1] / a_b_count[2]

    def is_color_dataset(config):
        color = None
        for ds_conf in config["dataset_list"]:
            n_channels = get_channel_number(ds_conf["path"], ds_conf.get("channel_name", "raw"), ds_conf.get("keyword", None), n_spatial_dims=2)
            if n_channels not in [1, 3]:
                raise ValueError(f"dataset has {n_channels} channels but have either 1 or 3 (color)")
            if color is None:
                color = n_channels == 3
            elif color != (n_channels == 3):
                raise ValueError(f"at least one dataset has color images and one has grayscale images")
        return color

    def get_dataset_center_scale(config, dataset_type="TRAIN"):
        scaling_parameters = config["dataset_parameters"].get("scaling_parameters", {"mode":"MODE_PERCENTILE", "percentile":95})
        if scaling_parameters["mode"]=="CONSTANT":
            return [scaling_parameters["center"], scaling_parameters["scale"]]
        elif scaling_parameters["mode"]=="MODE_PERCENTILE":
            mode_percentile_count = []
            for ds_conf in config["dataset_list"]:
                if ds_conf.get("type", "TRAIN") == dataset_type:
                    mode_percentile_count.append(get_center_scale(ds_conf["path"], ds_conf.get("channel_name", "raw"), ds_conf.get("keyword", None), method="mode-percentile", return_count=True))
            return weighted_avg(mode_percentile_count)
        elif scaling_parameters["mode"]=="MEAN_SD":
            center_scale_count = []
            for ds_conf in config["dataset_list"]:
                if ds_conf.get("type", "TRAIN") == dataset_type:
                    center_scale_count.append(get_center_scale(ds_conf["path"], ds_conf.get("channel_name", "raw"), ds_conf.get("keyword", None), method="mean-sd", return_count=True))
            return weighted_avg(center_scale_count)
        else:
            raise ValueError(f"Invalid scaling mode: {scaling_parameters['mode']}")

    def init_iterator(ds_kwargs, step_number, dataset=None, dataset_type="TRAIN", **kwargs):
        if dataset is None:
            dataset = ds_kwargs["path"]
            memory_persistent = WORKERS > 1 and not args.test_data_augmentation and should_load_dataset_in_shm(dataset, mode=ds_kwargs.get("shared_memory", "auto"))
        else:
            memory_persistent = isinstance(dataset, MemoryIO)
        channel_name = ds_kwargs.get("channel_name", "raw")
        group_keyword = ds_kwargs.get("keyword", None)
        n_frames = CONFIG["model_architecture"].pop("n_frames", 0)
        center_scale = kwargs["center_scale"]
        if dataset_type == "TRAIN":
            batch_size = ds_kwargs["batch_size"]
            tiling_parameters = ds_kwargs["tiling_parameters"]
            tiling_parameters["augmentation_rotate"] = NOISE_CORRELATION_RANGE is None
            tiling_parameters["perform_augmentation"] = True
            tiling_parameters["random_stride"] = True
            tiling_parameters["zoom_range"] = [1, 1]
            tiling_parameters["aspect_ratio_range"] = [1, 1]
            tiling_parameters["zoom_probability"] = 0
            extract_tiles_fun = extract_tile_random_zoom_function(**tiling_parameters)
            train_iterator = get_train_iterator(dataset, extract_tiles_fun=extract_tiles_fun,
                                                channel_keyword=channel_name, train_group_keyword=group_keyword,
                                                center_scale=center_scale,
                                                n_frames=n_frames,
                                                mask=True, # not RENOISE_MODE
                                                step_number=step_number, batch_size=batch_size, memory_persistent=memory_persistent, shuffle=kwargs.get("shuffle", True))
            if RENOISE_MODE:
                collapse_test_iterator = get_collapse_test_iterator(dataset, channel_keyword=channel_name, step_number=2, group_keyword=group_keyword, n_frames=n_frames)
                return train_iterator, collapse_test_iterator
            else:
                return train_iterator

        elif dataset_type == "EVAL":
            if CONFIG["model_architecture"]["architecture_type"] == "UNetMultiFrame":
                n_downsampling = len(CONFIG["model_architecture"]["encoder_settings"])
            else:
                n_downsampling = CONFIG["model_architecture"].get("n_downsampling", 3)
            contraction_factor = 2**n_downsampling
            return get_eval_iterator(dataset, noisy_channel=channel_name, group_keyword=group_keyword, n_frames=n_frames, contraction_factor=contraction_factor, memory_persistent=memory_persistent)


    def init_model():
        arch_args = copy.deepcopy(CONFIG["model_architecture"])
        n_frames = arch_args.pop("n_frames", 0)
        if COLOR and n_frames > 0 :
            raise ValueError("multiple frame is incompatible with color dataset")
        n_components = arch_args.pop("n_components", 3 if RENOISE_MODE else 1)
        dark_noise_sigma = arch_args.pop("dark_noise_sigma", 0)
        arch_type = arch_args.pop("architecture_type", "unetmultiframe").lower()
        if arch_type=="unetmultiframe":
            dnet = get_dnet_multiframe(n_frames=n_frames if not COLOR else 1, **arch_args)
        elif arch_type=="unet":
            n_filters = arch_args.get("filters", 96)
            depth = arch_args.get("n_downsampling", 3)
            skip = arch_args.get("skip_connection_mode", "NORMAL").lower()
            n_conv1x1 = arch_args.get("n_tail_conv", 2)
            dnet = get_dnet(n_filters=n_filters, depth = depth, skip_sg = skip=="stop_gradient", skip_omit=[0] if skip=="omit" else None, n_conv1x1=n_conv1x1, input_channels=3 if COLOR else 2 * n_frames + 1)
        elif arch_type == "unetn2n":
            depth = arch_args.pop("n_downsampling", 3)
            dnet = get_dnet_n2n(depth=depth, input_channels=3 if COLOR else 2 * n_frames + 1, **arch_args)
        else:
            raise ValueError(f"Unknown architecture: {arch_type}")
        denoiser = BlindDenoiser(n_components, basename=MODEL_NAME, dnet=dnet,
                                 convolution=get_convolution(PSF), renoise_correlation_range=NOISE_CORRELATION_RANGE,
                                 train_on_central_channel_only=False, dark_noise_sigma=dark_noise_sigma)
        denoiser.flip_invariance_transpose = False

        if args.export_only or args.compute_metrics and os.path.exists(WEIGHT_PATH):
            assert os.path.exists(WEIGHT_PATH), f"weights {WEIGHT_PATH} not found"
            denoiser.load_weights(WEIGHT_PATH)
            print(f"Weights loaded : {WEIGHT_PATH}", flush=True)
        elif LOAD_WEIGHT_PATH is not None or args.compute_metrics:
            assert os.path.exists(LOAD_WEIGHT_PATH), f"weights {LOAD_WEIGHT_PATH} not found"
            if os.path.isdir(LOAD_WEIGHT_PATH):
                loaded_model = tf.keras.models.load_model(LOAD_WEIGHT_PATH)
                denoiser.set_weights(loaded_model.get_weights())
            else:
                denoiser.load_weights(LOAD_WEIGHT_PATH)
            print(f"Weights loaded : {LOAD_WEIGHT_PATH}", flush=True)
        return denoiser

    def export_model(denoiser, path, avg_flip:bool=False):
        if avg_flip:
            if NOISE_CORRELATION_RANGE is None: # rotation allowed
                denoiser.set_flip_invariance(True, True, 1)  # if Y and X have different shapes, tensors cannot be concatenated -> n_flip_per_batch=1
            else: # rotation forbidden
                denoiser.set_flip_invariance(True, False, 1)
        else:
            denoiser.set_flip_invariance(False, False, 1)
        tf.saved_model.save(denoiser.get_inference_model(central_output_channel=True), path)

    COLOR = is_color_dataset(CONFIG)
    if args.export_only:
        print(f"export only: init model with weights: {WEIGHT_PATH} (exist: {os.path.exists(WEIGHT_PATH)})", flush=True)
        denoiser = init_model()
        export_model(denoiser, SAVED_MODEL_PATH)
        print("model saved", flush=True)
    else:
        print(f"init iterator...", flush=True)
        # compute scaling
        CENTER_SCALE = get_dataset_center_scale(CONFIG, dataset_type="TRAIN")
        if CENTER_SCALE is None and args.compute_metrics:
            CENTER_SCALE = get_dataset_center_scale(CONFIG, dataset_type="EVAL")
        print(f"Intensity normalization: center={CENTER_SCALE[0]} scale={CENTER_SCALE[1]}")
        scaling_file = os.path.join(args.config_dir, "train_scaling_parameters.json")
        with open(scaling_file, 'w') as f:
            f.write(f'{{"center":{CENTER_SCALE[0]}, "scale":{CENTER_SCALE[1]}}}')
        test_it = None
        if args.test_data_augmentation:
            train_it = get_iterator(CONFIG, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE, center_scale=CENTER_SCALE)
            test_param = CONFIG.get("test_data_augmentation_parameters", {})
            input_only = test_param.get("input_only", True)
            if RENOISE_MODE:
                train_it = train_it[0]
                #input_only = True
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
                input = train_it[idx]
                if True: # not RENOISE_MODE
                    input, output = input
                    if not input_only:
                        outputs.append(output)
                inputs.append(input)
                print(f"{i + 1}/{n_iterations}", flush=True)
            train_it.close()
            input = np.stack(inputs, 0)
            transpose_axis = [4, 0, 1, 2, 3]
            input = np.transpose(input, transpose_axis)
            if not input_only:
                output = np.stack(outputs, 0)
                output = np.transpose(output, transpose_axis)
                output_name = "MASKED"
            print(f"writing {1 + (0 if input_only else 1)} x {input.shape} to file: {file_path}", flush=True)
            with h5py.File(file_path, mode='w') as h5pyFile :
                h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/input", data=input)
                if not input_only:
                    h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/output_0_{output_name}", data=output)


        elif args.compute_metrics:
            raise ValueError("Not supported yet")
        else: # training
            train_it = get_iterator(CONFIG, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE, dataset_type="TRAIN", center_scale=CENTER_SCALE)
            if RENOISE_MODE:
                train_it, collapse_test_it = train_it
                if is_list(collapse_test_it): # collapse test performed on first iterator only
                    collapse_test_it = collapse_test_it[0]
            else:
                collapse_test_it = None
            eval_iterator = get_iterator(CONFIG, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE, dataset_type="EVAL", center_scale=CENTER_SCALE)
            # init model
            print("init model...", flush=True)
            denoiser = init_model()
            print(f"it[0]: {train_it[0][0].shape}, {train_it[0][1].shape}; channels: {denoiser.input_channels}, {denoiser.input_channels}", flush=True)
            N_EPOCHS -= START_EPOCH
            if N_EPOCHS > 0: # perform training
                train_data = train_denoiser(denoiser, train_it,
                                            n_epochs=N_EPOCHS, start_epoch=START_EPOCH, step_number = STEP_NUMBER,
                                            n_epochs_masked_training=0,
                                            training_mode=1 if RENOISE_MODE else 0,
                                            learning_rate=LR, learning_rate_min=MIN_LR, epsilon=EPSILON_RANGE[0], epsilon_min=EPSILON_RANGE[1],
                                            collapse_test_iterator=collapse_test_it, collapse_test_n_max=0,
                                            eval_iterator=eval_iterator, eval_center_scale=CENTER_SCALE, eval_data_range=None, eval_period=1, eval_verbose=0,
                                            weight_path=WEIGHT_PATH, log_path=LOG_PATH, additional_callbacks=None,
                                            fit_kwargs={"workers": WORKERS})
            elif START_EPOCH > 0:
                print("Start Epoch is greater than Epoch number.", flush=True)

            if not args.train_only: # export model
                print("saving model...", flush=True)
                export_model(denoiser, SAVED_MODEL_PATH)
                print("model saved", flush=True)
