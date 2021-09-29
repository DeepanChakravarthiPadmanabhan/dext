import numpy as np
from sklearn.decomposition import PCA


def normalize_saliency_map(saliency_map):
    vmax = np.max(saliency_map)
    vmin = np.min(saliency_map)
    new_saliency = np.clip((saliency_map - vmin) / (vmax - vmin), 0, 1)
    return new_saliency


def add_saliency_maps(saliency_list):
    new_saliency = sum(saliency_list)
    new_saliency = normalize_saliency_map(new_saliency)
    return new_saliency


def average_saliency_maps(saliency_list):
    new_saliency = sum(saliency_list)
    new_saliency = new_saliency / len(saliency_list)
    new_saliency = normalize_saliency_map(new_saliency)
    return new_saliency


def get_and_mask(saliency):
    and_mask = np.logical_and.reduce(saliency)
    # TODO: Correct below condition check according to PEP8
    and_mask[and_mask == True] = 1
    and_mask[and_mask == False] = 0
    and_mask = and_mask.astype('float32')
    return and_mask


def and_saliency_maps(saliency_list):
    saliency = np.array(saliency_list)
    and_mask = get_and_mask(saliency)
    new_saliency = []
    for n, i in enumerate(saliency_list):
        modified = np.multiply(i, and_mask)
        new_saliency.append(modified)
    return new_saliency


def and_add_saliency_maps(saliency_list):
    new_saliency = and_saliency_maps(saliency_list)
    new_saliency = add_saliency_maps(new_saliency)
    return new_saliency


def and_average_saliency_maps(saliency_list):
    new_saliency = and_saliency_maps(saliency_list)
    new_saliency = average_saliency_maps(new_saliency)
    return new_saliency


def pca_saliency_maps(saliency_list):
    saliency = np.array(saliency_list)
    saliency = np.transpose(saliency, [1, 2, 0])
    saliency = np.reshape(saliency, (-1, len(saliency_list)))
    new_saliency = PCA(n_components=1).fit_transform(saliency)
    new_saliency = np.reshape(np.squeeze(new_saliency, axis=-1), (512, 512))
    new_saliency = normalize_saliency_map(new_saliency)
    return new_saliency


def merge_saliency(saliency_list, merge_method='add'):
    if merge_method == 'or_add':
        return add_saliency_maps(saliency_list)
    elif merge_method == 'and_add':
        return and_add_saliency_maps(saliency_list)
    elif merge_method == 'or_average':
        return average_saliency_maps(saliency_list)
    elif merge_method == 'and_average':
        return and_average_saliency_maps(saliency_list)
    elif merge_method == 'pca':
        return pca_saliency_maps(saliency_list)
    else:
        raise ValueError("Saliency merge method not implemented %s"
                         % merge_method)
