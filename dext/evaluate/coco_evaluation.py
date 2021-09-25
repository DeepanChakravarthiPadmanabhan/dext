from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json


def get_coco_metrics(result_file=None, annotation_file=None):
    if not result_file:
        result_file = 'eval.json'
    if not annotation_file:
        dataset_path = "/media/deepan/externaldrive1/datasets_project_repos/"
        dataset_name = "coco"
        data_dir = dataset_path + dataset_name
        annotation_file = data_dir + '/annotations/instances_val2017.json'

    cocoGt = COCO(annotation_file)
    cocoDt = cocoGt.loadRes(result_file)
    detections = json.load(open(result_file, 'r'))
    imgIds = [img_ids['image_id'] for img_ids in detections]
    imgIds = sorted(list(set(imgIds)))
    del detections

    # running evaluation
    # https://www.programmersought.com/article/3065285708/
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()
    return cocoEval.stats
