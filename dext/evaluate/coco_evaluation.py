from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import json

res_file = 'eval.json'
dataset_path = "/media/deepan/externaldrive1/datasets_project_repos/"
dataset_folder = "coco"
data_dir = dataset_path + dataset_folder
annotation_file = data_dir + '/annotations/instances_val2017.json'

cocoGt = COCO(annotation_file)
cocoDt = cocoGt.loadRes(res_file)
detections = json.load(open(res_file, 'r'))
imgIds = [img_ids['image_id'] for img_ids in detections]
imgIds = sorted(list(set(imgIds)))
del detections

# running evaluation
cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
cocoEval.params.imgIds = imgIds
cocoEval.evaluate()
cocoEval.accumulate()
cocoEval.summarize()