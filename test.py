from training_scripts.training_core import open_config_file
from training_scripts.training_distnet2d import init_iterator

path = "/data/DL/DistNet2D/Maxime/Test"
config = open_config_file(path)
ds = config["dataset_list"][0]

it = init_iterator(100, **ds)

print("it: init")
print(f"it len {len(it)}")
print(f"it 0 {it[0].shape}")
