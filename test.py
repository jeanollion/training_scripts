from pathlib import Path
import sys
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

path_root = Path(__file__).parents[2] / "training_scripts"
#sys.path.append(str(path_root ))
print(path_root)

from training_scripts.training_core import open_config_file, get_iterator
print("test")
import argparse
import os
import random
import numpy as np
import tensorflow as tf
import h5py
from dataset_iterator.image_data_generator import get_image_data_generator
from pix_mclass.utils import ensure_multiplicity
from pix_mclass import get_unet
import pix_mclass.training as pmt

print(np.tile(np.arange(10), 2))