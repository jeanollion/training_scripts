import argparse
import os, resource
import sys
import random
import numpy as np
import tensorflow as tf
import h5py
from importlib.metadata import version
from dataset_iterator.image_data_generator import get_image_data_generator
from dataset_iterator.datasetIO import MemoryIO
from dataset_iterator.ordered_enqueuer_cf import OrderedEnqueuerCF
from dataset_iterator.keras_callbacks import EpsilonCosineDecayCallback, LogsCallback, SafeModelCheckpoint
from pix_mclass.utils import ensure_multiplicity
from pix_mclass import get_unet
from pix_mclass.losses import get_class_weights, weighted_sparse_categorical_crossentropy
import pix_mclass.training as pmt
from training_core import open_config_file, get_iterator, should_load_dataset_in_shm, get_shm_info

__VERSION__ = "1.1.0"

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--export_only", action="store_true", help="skip model training and export model")
parser.add_argument("--test_data_augmentation", action="store_true", help="generate and store example of augmented data")
parser.add_argument("--class_number", type=int, default=3, help="number of class to predict (only used in export_only mode)")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
parser.add_argument("--n_epochs", type=int, help="number of training epochs")
parser.add_argument("--step_number", type=int, help="number of training steps per epoch")
parser.add_argument("--patience", type=int, help="patience for learning rate decrease during training")
parser.add_argument("--learning_rate", type=float, help="initial learning rate for training")
parser.add_argument("--min_learning_rate", type=float, help="minimal learning rate for training")

if __name__ == "__main__":
    args = parser.parse_args()

    # get parameters
    config = open_config_file(args.config_dir, args.test_data_augmentation)
    t_p = config["training_parameters"]
    model_name = t_p["model_name"] + (f"_{args.model_idx}" if args.model_idx is not None else "")
    WEIGHT_PATH = os.path.join(args.config_dir, t_p["weight_dir"],  model_name + ".h5") if len(t_p["weight_dir"])>0 else os.path.join(args.config_dir,  model_name + ".h5")
    LOAD_WEIGHT_PATH = t_p["load_model_file"] if len(t_p.get("load_model_file", "")) > 0 else None
    LOG_PATH = os.path.join(args.config_dir, t_p["log_dir"], model_name ) if len(t_p["log_dir"])>0 else os.path.join(args.config_dir, model_name )
    SAVED_MODEL_PATH = os.path.join(args.export_dir if args.export_dir is not None else args.config_dir, model_name)
    N_EPOCHS = args.n_epochs if args.n_epochs is not None else t_p.get("n_epochs", 500)
    STEP_NUMBER = args.step_number if args.step_number is not None else t_p.get("step_number", 200)
    VAL_STEP_NUMBER = t_p.get("validation_step_number", 100)
    PATIENCE = args.patience if args.patience is not None else t_p.get("patience", 40)
    LR = args.learning_rate if args.learning_rate is not None else t_p.get("learning_rate", 2e-4)
    MIN_LR = args.min_learning_rate if args.min_learning_rate is not None else t_p.get("min_learning_rate", 5e-7)
    EPSILON_RANGE = t_p.get("epsilon_range", [0.1, 1e-7])
    EPSILON_RANGE = [max(EPSILON_RANGE), min(EPSILON_RANGE)]
    WORKERS = min(os.cpu_count(), t_p.get("multiprocessing_workers", 1))
    SHUFFLE = not args.test_data_augmentation
    START_EPOCH = t_p.get("epoch_start", 0)

    print(f"Script version: {__VERSION__}; dataset_iterator version: {version('dataset_iterator')}; PixMClass version: {version('PixMClass')}")
    print(f"configuration file found. ")

    def init_iterator(ds_conf, step_number, dataset=None, dataset_type="TRAIN", **kwargs):
        data_aug_params = ds_conf.get("data_augmentation", {})
        channel_names = ds_conf.get("channel_name", "raw")
        if not isinstance(channel_names, (list, tuple)):
            channel_names = [channel_names]
        classes_name = ds_conf.get("classes_name", "classes")
        if dataset is None:
            dataset = ds_conf["path"]
            memory_persistent = WORKERS > 1 and not args.test_data_augmentation and should_load_dataset_in_shm(dataset, mode=ds_conf.get("shared_memory", "auto"))
        else:
            memory_persistent = isinstance(dataset, MemoryIO)
        if dataset_type=="TRAIN":
            weights = get_class_weights(dataset, classes_name) # inverse frequency
            weight_limit = ds_conf.get("loss_weight_range", None)
            if weight_limit is not None:
                assert len(weight_limit) == 2, "Weight limit should be of length 2"
                weights = np.minimum(weights, np.max(weight_limit))
                weights = np.maximum(weights, np.min(weight_limit))
        scaling_parameters = data_aug_params.get("scaling_parameters", None)
        if scaling_parameters is not None:
            scaling_parameters = ensure_multiplicity(len(channel_names), scaling_parameters)
            for i, sp in enumerate(scaling_parameters):
                sp["dataset"] = dataset
                sp["channel_name"] = channel_names[i]
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
                                tiling_parameters=tiling_parameters, batch_size=batch_size, step_number=step_number, dtype="float32", shuffle=kwargs.get("shuffle", True),
                                elasticdeform_parameters=data_aug_params.get("elasticdeform_parameters", None)
                                )
        if dataset_type=="TRAIN":
            return it, weights
        else:
            return it

    def init_model(n_classes):
        model = get_unet(n_classes, skip_omit=0)
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
        model = init_model(args.class_number)
        assert os.path.exists(WEIGHT_PATH), f"weights {WEIGHT_PATH} not found"
        model.load_weights(WEIGHT_PATH)
        # export model
        tf.saved_model.save(model, SAVED_MODEL_PATH)
        print("model saved", flush=True)
    else:
        print(f"init iterator...", flush=True)
        train_it, weight_list = get_iterator(config, init_iterator, step_number=STEP_NUMBER, shuffle=SHUFFLE, dataset_type="TRAIN")
        test_it = get_iterator(config, init_iterator, step_number=VAL_STEP_NUMBER, shuffle=SHUFFLE, dataset_type="TEST")
        if len(weight_list) > 1:
            # weighted sum of weights
            weights = np.zeros_like(weight_list[0])
            tot = 0
            for it, w in zip(train_it.iterators, weight_list):
                l = it.get_sample_number() # usually differs from len(it)
                tot += l
                weights += w * l
            weights /= tot
        else:
            weights = weight_list[0]
        print(f"Class weights: {weights}", flush=True)

        if args.test_data_augmentation:
            test_param = config.get("test_data_augmentation_parameters", {})
            input_only = test_param.get("input_only", True)
            n_iterations = test_param.get("iteration_number", 50)
            root_path = "/dataTemp" if os.path.exists("/dataTemp") else "/data"
            file_path = os.path.join(root_path, "test_data_augmentation.h5")

            print(f"generating data augmented images : n_iterations: {n_iterations} output file: {file_path} ...", flush=True)
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
            input = np.stack(inputs, 1)
            transpose_axis = [0, 1, 4, 2, 3]
            input = np.transpose(input, transpose_axis)
            if not input_only:
                output = np.stack(outputs, 1)
                output = np.transpose(output, transpose_axis)
            print(f"writing {len(outputs) + 1 } x {input.shape} to file: {file_path}", flush=True)
            with h5py.File(file_path, mode='w') as h5pyFile :
                h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/input", data=input)
                if not input_only:
                    h5pyFile.create_dataset(f"data_aug/batch_idx{idx}/output", data=output)
            if (os.path.exists("/dataTemp")):
                print(f"dataTemp exists ! {os.listdir('/dataTemp')}", flush=True)
        else:
            # init model
            print("init model...", flush=True)
            loss = weighted_sparse_categorical_crossentropy(weights, dtype="float32")
            model = init_model(weights.shape[0])
            model.compile(optimizer=tf.keras.optimizers.Adam(LR, epsilon=EPSILON_RANGE[0]), loss=loss)

            # perform training
            checkpoint = SafeModelCheckpoint(WEIGHT_PATH, monitor='val_loss' if test_it is not None else 'loss', verbose=1, save_best_only=True, save_weights_only=True)
            lr_schedule = tf.keras.callbacks.ReduceLROnPlateau(min_lr=MIN_LR, factor=0.5, patience=PATIENCE, verbose=1, min_delta=0.001, monitor='val_loss' if test_it is not None else 'loss')
            ton_cb = tf.keras.callbacks.TerminateOnNaN()
            log_cb = LogsCallback(LOG_PATH + ".csv", start_epoch=START_EPOCH)
            callbacks = [checkpoint, lr_schedule, log_cb, ton_cb]
            if EPSILON_RANGE[1]!=EPSILON_RANGE[0]:
                eps_schedule = EpsilonCosineDecayCallback(decay_steps=N_EPOCHS * STEP_NUMBER, start_epsilon=EPSILON_RANGE[0],  min_epsilon=EPSILON_RANGE[1], start_step=START_EPOCH * STEP_NUMBER, verbose=1)
                callbacks.append(eps_schedule)
            print("start training...", flush=True)
            N_EPOCHS -= START_EPOCH
            if N_EPOCHS > 0:
                train_it.open()
                if WORKERS > 1:
                    # check available shm:
                    shm = get_shm_info(verbose=2)
                    if shm is not None and shm[2] < 1:
                        print(f"Warning: available shared memory is low: {shm[2]:.2f}/{shm[0]:.2f}G, this can hamper multiprocessing", flush=True)
                    #enq = tf.keras.utils.OrderedEnqueuer(train_it, use_multiprocessing=True, shuffle=True)
                    enq = OrderedEnqueuerCF(train_it, shuffle=True) #, name="train_gen"
                    enq.start(workers=WORKERS, max_queue_size=max(3, min(STEP_NUMBER, WORKERS)))
                    gen = enq.get()
                    if test_it is not None: # TODO syncronize test gen and train gen
                        test_enq = OrderedEnqueuerCF(test_it, shuffle=False) #, name="test_gen"
                        test_enq.start(workers=WORKERS, max_queue_size=max(3, min(STEP_NUMBER, WORKERS)))
                        test_gen = test_enq.get()
                    else:
                        test_gen = None
                else:
                    gen = train_it
                    test_gen = test_it
                model.fit(gen, epochs=N_EPOCHS, steps_per_epoch=STEP_NUMBER, validation_data=test_gen, callbacks=callbacks, validation_steps=VAL_STEP_NUMBER, validation_freq=t_p.get("validation_frequency", 1))
                if WORKERS > 1:
                    print("stopping enqueuer", flush=True)
                    enq.stop()
                    if test_it is not None:
                        test_enq.stop()
                print("end of training", flush=True)
            elif START_EPOCH > 0:
                print("Start Epoch is greater than Epoch number.", flush=True)
            train_it.close()
            if test_it is not None:
                test_it.close()
            if not args.train_only: # export model
                print("saving model...", flush=True)
                tf.saved_model.save(model, SAVED_MODEL_PATH)
                print("model saved", flush=True)
