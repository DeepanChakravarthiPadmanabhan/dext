import logging
import json
from paz.datasets.utils import get_class_names
from paz.processors.image import LoadImage
from paz.models.detection.ssd512 import SSD512
from dext.model.ssd.utils import ssd_preprocess
from dext.model.ssd.ssd_postprocess import ssd_postprocess
from dext.dataset.coco import COCODataset
from dext.evaluate.utils import get_evaluation_details


LOGGER = logging.getLogger(__name__)


dataset_path = "/media/deepan/externaldrive1/datasets_project_repos/"
dataset_folder = "coco"
data_dir = dataset_path + dataset_folder
eval_images = COCODataset(data_dir, "val", name="val2017")
datasets = eval_images.load_data()
LOGGER.info('Number of images in the dataset %s' % len(datasets))

class_names = get_class_names("COCO")
num_classes = len(class_names)
loader = LoadImage()

model = SSD512(num_classes=num_classes, weights="COCO")
preprocessing = ssd_preprocess
postprocessing = ssd_postprocess

eval_json = []
for n, data in enumerate(datasets):
    LOGGER.info('Evaluating on image: %s' % n)
    image_path = data['image']
    image_index = data['image_index']
    raw_image = loader(image_path)
    image = raw_image.copy()
    input_image, image_scales = preprocessing(image, model.input_shape[1:3])
    outputs = model(input_image)
    detection_image, detections, class_map_idx = postprocessing(
    model, outputs, image_scales, raw_image.copy())
    # plt.imsave("ssd512" + str(n) + ".jpg", detection_image)
    # LOGGER.info('Saved entry: %s' % n)
    all_boxes = get_evaluation_details(detections)
    for i in all_boxes:
        eval_entry = {'image_id': image_index, 'category_id': i[5],
                      'bbox': i[:4], 'score': i[4]}
        eval_json.append(eval_entry)
    LOGGER.info('Added json entry: %s' % n)
with open('eval.json', 'w', encoding='utf-8') as f:
    json.dump(eval_json, f, ensure_ascii=False, indent=4)

