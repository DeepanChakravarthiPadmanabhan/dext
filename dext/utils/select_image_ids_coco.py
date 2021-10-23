import os
import gin
import pandas as pd
import numpy as np


@gin.configurable
def filter_image_ids(results_dir):
    if os.path.exists(results_dir):
        files = os.listdir(results_dir)
        excel_files = [i for i in files if '.xlsx' in i]
        ran_ids = []
        for i in excel_files:
            path = os.path.join(results_dir, i)
            reg_error_sheet = pd.read_excel(path, sheet_name='reg_error_curve')
            ran_ids.extend(list(reg_error_sheet['image_index'].unique()))
        ran_ids = list(np.unique(np.array(ran_ids)))
        return ran_ids
    else:
        raise ValueError('Results directory not found.')

