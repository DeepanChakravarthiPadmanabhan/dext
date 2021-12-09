import logging

from dext.utils.class_names import get_classes

LOGGER = logging.getLogger(__name__)


def find_errors(gt, detections, dataset_name, model_name):
    print('GT: ', gt)
    print('DET: ', detections)
    class_names = get_classes(dataset_name, model_name)
    for i in gt:
        class_info = class_names[int(i[-1])]
    return 0, 0, 0, 0
