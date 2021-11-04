import os
import gin
import json
import numpy as np


@gin.configurable
def filter_image_ids(results_dir):
    if os.path.exists(results_dir):
        file = os.path.join(results_dir, 'saliency_image_paths')
        data = [json.loads(line) for line in open(file, 'r')]
        data = np.array(data)
        image_index = data[:, 0]
        ran_ids = list(np.unique(image_index))
        ran_ids = [int(i) for i in ran_ids]
        return ran_ids
    else:
        raise ValueError('Results directory not found.')

