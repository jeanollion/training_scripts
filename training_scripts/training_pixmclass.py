import argparse
import os, resource
import shutil
import sys
import random
import time

import numpy as np
import tensorflow as tf
import h5py
from importlib.metadata import version
from tensorflow.keras import mixed_precision
from dataset_iterator.validation_callback import ValidationCallback
from dataset_iterator.image_data_generator import get_image_data_generator
from dataset_iterator.datasetIO import MemoryIO
from dataset_iterator.nonvoid_iterator import NonVoidIterator
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF
from dataset_iterator.keras_callbacks import EpsilonCosineDecayCallback, LogsCallback, SafeModelCheckpoint, \
    LogLRCallback
from pix_mclass.callbacks import CosineDecayResume
from pix_mclass.unet import get_model
from pix_mclass.utils import ensure_multiplicity
from pix_mclass.losses import get_class_counts, get_weighted_sparse_categorical_crossentropy, \
    get_weighted_sparse_categorical_tempered_focal_loss
import pix_mclass.training as pmt
from training_core import open_config_file, get_iterator, should_load_dataset_in_shm, get_shm_info, check_requirements, \
    compare_versions, print_requirement_error, export_fp16_model

__VERSION__ = "1.1.5"
__REQUIRES__ = ["dataset_iterator>=0.5.8", "PixMClass>=0.1.6" ]

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--train_only", action="store_true", help="train but no export")
parser.add_argument("--export_only", action="store_true", help="skip model training and export model")
parser.add_argument("--test_data_augmentation", action="store_true", help="generate and store example of augmented data")
parser.add_argument("--test_predict", action="store_true", help="make predictions on evaluation dataset")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
parser.add_argument("--n_epochs", type=int, help="number of training epochs")
parser.add_argument("--step_number", type=int, help="number of training steps per epoch")
parser.add_argument("--patience", type=int, help="patience for learning rate decrease during training")
parser.add_argument("--learning_rate", type=float, help="initial learning rate for training")
parser.add_argument("--min_learning_rate", type=float, help="minimal learning rate for training")
parser.add_argument("--strategy", default="", type=str, help="distributed training strategy: multiworker-slurm or mirrored. Leave empty for default behaviour (single replica)")
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
            print(f"script version is out-of-date: {__VERSION__} minimal version: {args.min_script_version}",
                  flush=True)
            print_requirement_error()
            sys.exit(1)
    print( f"Script version: {__VERSION__}; dataset_iterator version: {version('dataset_iterator')}; PixMClass version: {version('PixMClass')}")
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
    VAL_FREQ = validation_freq=t_p.get("validation_frequency", 1)
    PATIENCE = args.patience if args.patience is not None else t_p.get("patience", 40)
    LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
    MIN_LR = args.min_learning_rate if args.min_learning_rate is not None else t_p.get("min_learning_rate", 5e-7)
    EPSILON_RANGE = t_p.get("epsilon_range", [0.1, 1e-7])
    EPSILON_RANGE = [max(EPSILON_RANGE), min(EPSILON_RANGE)]
    if args.strategy == "multiworker-slurm":
        WORKERS = int(os.environ.get("SLURM_CPUS_PER_TASK", 1))
    else:
        WORKERS = t_p.get("multiprocessing_workers", 1)
    WORKERS = min(os.cpu_count(), WORKERS)
    SHUFFLE = not RUN_TEST
    START_EPOCH = t_p.get("start_epoch", 0)
    TRIDIMENSIONAL_MODE = len(config.get("dataset_parameters", {}).get("input_shape", [None, None])) == 3
    print(f"configuration file found. ")

    def init_iterator(ds_conf, step_number, dataset=None, dataset_type="TRAIN", **kwargs):
        data_aug_params = ds_conf.get("data_augmentation", {})
        channel_names = ds_conf.get("channel_name", "raw")
        if not isinstance(channel_names, (list, tuple)):
            channel_names = [channel_names]
        classes_name = ds_conf.get("classes_name", "classes")
        if dataset is None:
            dataset = ds_conf["path"]
            memory_persistent = WORKERS > 1 and not RUN_TEST and should_load_dataset_in_shm(dataset, mode=ds_conf.get("shared_memory", "auto"))
        else:
            memory_persistent = isinstance(dataset, MemoryIO)
        if dataset_type=="TRAIN":
            class_counts, bck_count = get_class_counts(dataset, classes_name)
        scaling_parameters = data_aug_params.get("scaling_parameters", None)
        if scaling_parameters is not None:
            scaling_parameters = ensure_multiplicity(len(channel_names), scaling_parameters)
            for i, sp in enumerate(scaling_parameters):
                sp["dataset"] = dataset
                sp["channel_name"] = channel_names[i]
                sp["group_keyword"] = ds_conf.get("keyword", None)
        else:
            scaling_parameters = [{}]*len(channel_names)
        scaling_data_generators = [get_image_data_generator(scaling_parameters=scaling_parameters[i]) for i in range(len(channel_names))]

        illumination_parameters = data_aug_params.get("illumination_parameters", None)
        if illumination_parameters is not None:
            illumination_generator = get_image_data_generator(illumination_parameters=illumination_parameters)
        else:
            illumination_generator = None

        batch_size = ds_conf["batch_size"]
        tiling_parameters = ds_conf.get("tiling_parameters", None)

        it = pmt.get_iterator(dataset, memory_persistent=memory_persistent, scaling_data_generator=scaling_data_generators, illumination_data_generator=illumination_generator,
                                input_channel_keywords=channel_names, class_keyword=classes_name,
                                train_group_keyword=ds_conf.get("keyword", None),
                                tiling_parameters=tiling_parameters, batch_size=batch_size, step_number=step_number,
                                tridimensional_mode = TRIDIMENSIONAL_MODE,
                                dtype="float32", shuffle=kwargs.get("shuffle", True),
                                elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None)
                                )
        if dataset_type=="TRAIN":
            min_annotated_pixel_number = ds_conf.get("min_annotated_pixel_number", 0)
            if min_annotated_pixel_number > 0:
                it = NonVoidIterator(it, 0, False, pix_thld=min_annotated_pixel_number)
            return it, class_counts, bck_count
        else:
            return it

    def init_model(**kwargs):
        input_shape = config.get("dataset_parameters", {}).get("input_shape", [None, None])
        model = get_model(tridimensional_mode=TRIDIMENSIONAL_MODE, input_shape=input_shape, **kwargs)
        if args.export_only or (args.test_predict and os.path.exists(WEIGHT_PATH)):
            assert os.path.exists(WEIGHT_PATH), f"weights {WEIGHT_PATH} not found"
            model.load_weights(WEIGHT_PATH)
            print(f"Weights loaded : {WEIGHT_PATH}", flush=True)
        elif LOAD_WEIGHT_PATH is not None or args.test_predict:
            assert LOAD_WEIGHT_PATH is not None and os.path.exists(LOAD_WEIGHT_PATH), f"weights {LOAD_WEIGHT_PATH} not found"
            model.load_weights(LOAD_WEIGHT_PATH)
            print(f"Weights loaded : {LOAD_WEIGHT_PATH}", flush=True)
        return model

    def get_input_number(config):
        n = -1
        for ds_conf in config["dataset_list"]:
            channel_names = ds_conf.get("channel_name", "raw")
            cur_n = 1 if not isinstance(channel_names, (list, tuple)) else len(channel_names)
            if n<0:
                n=cur_n
            else:
                assert n == cur_n, f"invalid channel number for dataset: {ds_conf['path']}"
        return n

    N_INPUTS = get_input_number(config)
    arch_conf = config.get("model_architecture", {"architecture_type": "unet", "n_inputs": max(1, N_INPUTS), "n_classes": 3})
    if "n_classes" not in arch_conf:
        arch_conf["n_classes"] = 3
    if N_INPUTS == -1:
        N_INPUTS = arch_conf["n_inputs"]
    assert arch_conf["n_inputs"] == N_INPUTS, f"Inconsistent input number between datasets ({N_INPUTS}) and model {arch_conf['n_inputs']}"
    print(f"Input Number: {N_INPUTS}")

    if args.export_only:
        print(f"export only: init model with weights: {WEIGHT_PATH} (exist: {os.path.exists(WEIGHT_PATH)})")
        model = init_model(**arch_conf)
        # export model
        if args.export_fp16:
            export_fp16_model(model, SAVED_MODEL_PATH)
        else:
            model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True)
        print("model saved", flush=True)
    else:
        print(f"init iterator...", flush=True)
        train_it, class_count_list, bck_count_list = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE, dataset_type="TRAIN")

        if len(class_count_list) > 1:
            # weighted sum of weights
            class_counts = np.stack(class_count_list, 0)
            class_counts = np.sum(class_counts, axis=0, keepdims=False)
        else:
            class_counts = class_count_list[0]
        class_counts = np.maximum(class_counts, 1)
        n_classes = arch_conf.get("n_classes", 3)
        assert n_classes == class_counts.shape[0], f"dataset contains {class_counts.shape[0]} class, but model expects {n_classes} classes"

        loss_parameters =  config["training_parameters"].get("category_loss_parameters", {"weight_power_law":1, "focal_weight":0, "focal_weight_power_law": 0})
        inv_freq = np.mean(class_counts) / class_counts
        weights = np.power(inv_freq, loss_parameters.get("weight_power_law", 1))

        # annotation sparseness
        bck_count = np.sum(bck_count_list)
        fore_count = np.sum(class_counts)
        annotated_pix = float(fore_count) / float(bck_count + fore_count)
        lr_factor = (1 + annotated_pix) / 2
        print(f"Class counts: {class_counts}, class weights: {weights} annotated pixels: {annotated_pix * 100:.3}% corrected LR: {LR / lr_factor}", flush=True)

        if RUN_TEST:
            test_param = config.get("test_data_augmentation_parameters", {})
            input_only = test_param.get("input_only", True)
            n_iterations = test_param.get("iteration_number", 50)
            root_path = "/dataTemp" if os.path.exists("/dataTemp") else "/data"
            file_path = os.path.join(root_path, "test_data_augmentation.h5")

            print(f"generating data augmented images : n_iterations: {n_iterations} output file: {file_path} ...", flush=True)
            idx = test_param.get("batch_index", -1)
            if idx < 0 or idx >= len(train_it):
                idx = random.randint(0, len(train_it)-1)
            inputs = []
            outputs = []
            if args.test_data_augmentation:
                print(f"Generating {n_iterations} versions of sample {idx}", flush=True)
                for i in range(n_iterations):
                    input, output = train_it[idx]
                    if not isinstance(input, (tuple, list)):
                        input = [input]
                    inputs.append(input)
                    if not input_only:
                        outputs.append(output)
                    print(f"{i + 1}/{n_iterations}", flush=True)
            else: # test predict
                model = init_model(**arch_conf)
                model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON_RANGE[0]))
                input, _ = train_it[idx]
                if not isinstance(input, (tuple, list)):
                    input = [input]
                output = model.predict(input)
                inputs.append(input)
                outputs.append(output)

            transpose_axis = [0, 1, 5, 2, 3, 4] if TRIDIMENSIONAL_MODE else [0, 1, 4, 2, 3]
            input = []
            for i in range(N_INPUTS):
                print(f"input: {i+1}/{N_INPUTS} shape: {[in_[i].shape for in_ in inputs]}")
                local_input = np.stack([in_[i] for in_ in inputs], 1)
                local_input = np.transpose(local_input, transpose_axis)
                input.append(local_input)
            if not input_only:
                output = np.stack(outputs, 1)
                output = np.transpose(output, transpose_axis)
                if TRIDIMENSIONAL_MODE:
                    if args.test_data_augmentation:
                        output_mask = output[:, :, 1]
                        output = output[:, :, 0] # remove channel axis
                    else: # test predict
                        output = output[:, :, -1] # only last channel
            if TRIDIMENSIONAL_MODE: # remove channel axis
                input = [a[:, :, 0] for a in input]

            print(f"writing {len(outputs) + N_INPUTS } x {input[0].shape} to file: {file_path}", flush=True)
            with h5py.File(file_path, mode='w') as h5pyFile :
                for i in range(N_INPUTS):
                    h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/input{i}", data=input[i])
                if not input_only:
                    h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/output", data=output)
                    if TRIDIMENSIONAL_MODE and args.test_data_augmentation:
                        h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/output_mask", data=output_mask)
            if (os.path.exists("/dataTemp")):
                print(f"dataTemp exists ! {os.listdir('/dataTemp')}", flush=True)
        else:
            val_it = get_iterator(config, init_iterator, step_number=VAL_STEP_NUMBER, shuffle=SHUFFLE, dataset_type="TEST")

            # handling the current strategy
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
            loss_kwargs = {'reduction' : tf.losses.Reduction.NONE} if args.strategy != "" else {}
            #loss = get_weighted_sparse_categorical_crossentropy(weights, dtype="float32", **loss_kwargs)
            loss = get_weighted_sparse_categorical_tempered_focal_loss(weights, dtype="float32", temperature=loss_parameters.get("temperature", 0), pseudo_huber=loss_parameters.get("pseudo_huber", 0), label_smoothing=loss_parameters.get("label_smoothing", 0), focal_weight=loss_parameters.get("focal_weight", 0), **loss_kwargs)

            with strategy.scope():
                model = init_model(**arch_conf)
                corrected_lr = LR/lr_factor
                learning_rate = CosineDecayResume(initial_learning_rate=corrected_lr,
                                                  decay_steps=STEP_NUMBER * N_EPOCHS,
                                                  start_step=STEP_NUMBER * START_EPOCH,
                                                  alpha=float(MIN_LR) / float(corrected_lr),
                                                  warmup_learning_rate_factor=1. / 10,
                                                  warmup_steps=STEP_NUMBER * WARMUP_EPOCHS)
                model.compile(optimizer=tf.keras.optimizers.Adam(LR/lr_factor, epsilon=EPSILON_RANGE[0]), loss=loss)

            # perform training
            checkpoint = SafeModelCheckpoint(WEIGHT_PATH, monitor='val_loss' if val_it is not None and VAL_FREQ==1 else 'loss', verbose=1, save_best_only=True, save_weights_only=True)
            ton_cb = tf.keras.callbacks.TerminateOnNaN()
            log_cb = LogsCallback(LOG_PATH + ".csv", start_epoch=0)
            callbacks = [LogLRCallback(), checkpoint, log_cb, ton_cb]
            if EPSILON_RANGE[1]!=EPSILON_RANGE[0]:
                eps_schedule = EpsilonCosineDecayCallback(decay_steps=N_EPOCHS * STEP_NUMBER, start_epsilon=EPSILON_RANGE[0],  min_epsilon=EPSILON_RANGE[1], start_step=START_EPOCH * STEP_NUMBER, verbose=1)
                callbacks.append(eps_schedule)
            if val_it is not None:
                val_cb = ValidationCallback(val_it, STEP_NUMBER, validation_freq=VAL_FREQ, start_epoch=START_EPOCH)
                callbacks.append(val_cb)
            else:
                val_cb = None
            print("start training...", flush=True)
            if N_EPOCHS > START_EPOCH:
                train_it.open()
                if WORKERS > 1:
                    # check available shm:
                    shm = get_shm_info(verbose=2)
                    if shm is not None and shm[2] < 1:
                        print(f"Warning: available shared memory is low: {shm[2]:.2f}/{shm[0]:.2f}G, this can hamper multiprocessing", flush=True)
                    #enq = tf.keras.utils.OrderedEnqueuer(train_it, use_multiprocessing=True, shuffle=True)
                    enq = OrderedEnqueuerCF(train_it, shuffle=True) #, name="train_gen"
                    if val_it is not None:
                        val_enq = OrderedEnqueuerCF(val_it, shuffle=False, name="val", max_steps=VAL_STEP_NUMBER) #, name="test_gen"
                        val_cb.set_enqueuer(val_enq, enq)
                        val_enq.start(workers=WORKERS, max_queue_size=max(3, min(VAL_STEP_NUMBER, WORKERS)))
                        val_gen = val_enq.get(block=False, name="val")
                    else:
                        val_gen = None
                    enq.start(workers=WORKERS, max_queue_size=max(3, min(STEP_NUMBER, WORKERS)))
                    gen = enq.get()
                else:
                    gen = train_it
                    val_gen = val_it
                if val_cb is not None:
                    val_cb.initialize()
                model.fit(gen, epochs=N_EPOCHS, initial_epoch=START_EPOCH, steps_per_epoch=STEP_NUMBER, validation_data=val_gen, callbacks=callbacks, validation_steps=VAL_STEP_NUMBER, validation_freq=VAL_FREQ)
                if WORKERS > 1:
                    print("stopping enqueuer", flush=True)
                    enq.stop()
                    if val_it is not None:
                        val_enq.stop()
                print("end of training", flush=True)
            elif START_EPOCH > 0:
                print("Start Epoch is greater than Epoch number.", flush=True)
            train_it.close()
            if val_it is not None:
                val_it.close()
            if not args.train_only: # export model
                print("saving model...", flush=True)
                if args.strategy == "multiworker-slurm":
                    is_chief = (
                        cluster_resolver.task_type == "worker"
                        and cluster_resolver.task_id == 0
                    )
                    if is_chief:
                        model.load_weights(WEIGHT_PATH)  # reload best weights
                    save_path = (
                        SAVED_MODEL_PATH
                        if is_chief
                        else SAVED_MODEL_PATH + "_tmp_" + os.environ.get("SLURM_PROCID", "")
                    )
                    if args.export_fp16:
                        export_fp16_model(model, SAVED_MODEL_PATH)
                    else:
                        model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True)

                    print("model saved", flush=True)

                    if not is_chief:
                        print(f"cleaning temp models at {save_path}", flush=True)
                        shutil.rmtree(save_path)  # clean up for non chief worker
                else:
                    model.load_weights(WEIGHT_PATH)  # reload best weights
                    if args.export_fp16:
                        export_fp16_model(model, SAVED_MODEL_PATH)
                    else:
                        model.save(SAVED_MODEL_PATH, include_optimizer=False, save_traces=True)
                    print("model saved", flush=True)
