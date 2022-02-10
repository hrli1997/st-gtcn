import numpy as np

arry_path = "benchmarks/sns-lstm/datasets/navMap/"
arry_name = "hotel.npy"

x = np.load(arry_path+arry_name, mmap_mode='r')
print(x.shape, x)
