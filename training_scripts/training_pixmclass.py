import argparse
import os

parser = argparse.ArgumentParser()
parser.add_argument("config_dir", type=str, help="directory containing the configuration file")
parser.add_argument("--model_idx", type=int, help="index of model")
parser.add_argument("--continue_training", type=int, help="if specified, will load weight corresponding to model_idx before training and override them", action="store_true")
parser.add_argument("--export_dir", type=str, help="directory to export saved model to")
args = parser.parse_args()

t_p = config["training_parameters"]
model_name = t_p["model_name"]
weight_path = os.path.join(t_p["weight_dir"],  model_name + (f"_{args.model_idx}" if args.model_idx is not None else "") + ".h5")
log_path = os.path.join(t_p["log_dir"], model_name + (f"_{args.model_idx}" if args.model_idx is not None else ""))
saved_model_path = os.path.join(args.export_dir if args.export_dir else args.config_dir, model_name + (f"_{args.model_idx}" if args.model_idx is not None else ""))

N_EPOCHS = t_p["n_epochs"]
PATIENCE = t_p["patience"]