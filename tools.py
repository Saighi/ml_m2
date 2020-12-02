import numpy as np

def split(array_2d,nb):
    array_2d = np.array(np.split(array_2d, nb, axis=1))
    return array_2d.reshape(-1, array_2d.shape[-1])