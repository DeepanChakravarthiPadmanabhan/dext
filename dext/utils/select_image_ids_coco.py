import os
import gin
import json
import numpy as np


def get_history_file(results_dir, filename):
    if os.path.exists(results_dir):
        file = os.path.join(results_dir, filename)
        if os.path.exists(file):
            return file
        else:
            raise ValueError('%s file not found' % file)
    else:
        raise ValueError('Results directory not found.')


@gin.configurable
def filter_image_ids(results_dir, filename='saliency_image_paths'):
    file = get_history_file(results_dir, filename)
    data = [json.loads(line) for line in open(file, 'r')]
    data = np.array(data)
    image_index = data[:, 0]
    ran_ids = list(np.unique(image_index))
    ran_ids = [int(i) for i in ran_ids]
    return ran_ids


