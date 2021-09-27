import numpy as np
import matplotlib.pyplot as plt


def add_saliency_maps(saliency_list):
    new_saliency = sum(saliency_list)
    vmax = np.max(new_saliency)
    vmin = np.min(new_saliency)
    new_saliency = np.clip((new_saliency - vmin) / (vmax - vmin), 0, 1)
    return new_saliency


def merge_saliency(saliency_list, merge_method='add'):
    if merge_method == 'add':
        return add_saliency_maps(saliency_list)
    else:
        raise ValueError("Saliency merge method not implemented %s"
                         % merge_method)
