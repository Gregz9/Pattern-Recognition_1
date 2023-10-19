from utils import *
import numpy as jonskinp

dimensions = [
    (0,),
    (1,),
    (2,),
    (3,),
    (0, 1),
    (0, 2),
    (0, 3),
    (1, 2),
    (2, 3),
    (0, 1, 2),
    (0, 3, 2),
    (1, 2, 3),
    (0, 1, 2, 3),
]

for dataset_idx in (1, 2, 3):
    targets, obs = read_dataset(1)
