import os
import logging
import json
import matplotlib.pyplot as plt

from paz.processors.image import LoadImage
from dext.dataset.coco import COCODataset
from dext.evaluate.utils import get_evaluation_details
from dext.factory.preprocess_factory import PreprocessorFactory
from dext.factory.postprocess_factory import PostprocessorFactory
from dext.factory.inference_factory import InferenceFactory
from dext.explainer.utils import get_model
from dext.evaluate.coco_evaluation import get_coco_metrics


LOGGER = logging.getLogger(__name__)


def evaluate_model(model_name, image_size=512, dataset_path=None,
                   annotation_file=None, result_file=None):
    model = get_model(model_name)
    preprocessor_fn = PreprocessorFactory(model_name).factory()
    postprocessor_fn = PostprocessorFactory(model_name).factory()
    inference_fn = InferenceFactory(model_name).factory()
    eval_images = COCODataset(dataset_path, "val", name="val2017")
    datasets = eval_images.load_data()
    LOGGER.info('Number of images in the dataset %s' % len(datasets))
    eval_json = []
    for n, data in enumerate(datasets):
        LOGGER.info('Evaluating on image: %s' % n)
        image_path = data['image']
        image_index = data['image_index']
        loader = LoadImage()
        raw_image = loader(image_path)
        image = raw_image.copy()
        forward_pass_outs = inference_fn(
            model, image, preprocessor_fn,
            postprocessor_fn, image_size)
        detection_image = forward_pass_outs[0]
        detections = forward_pass_outs[1]
        plt.imsave(model_name + str(n) + ".jpg", detection_image)
        LOGGER.info('Saved entry: %s' % n)
        all_boxes = get_evaluation_details(detections)
        for i in all_boxes:
            eval_entry = {'image_id': image_index, 'category_id': i[4],
                          'bbox': i[:4], 'score': i[5]}
            eval_json.append(eval_entry)
        LOGGER.info('Added json entry: %s' % n)
    try:
        os.remove(result_file)
    except OSError:
        pass
    with open(result_file, 'w', encoding='utf-8') as f:
        json.dump(eval_json, f, ensure_ascii=False, indent=4)

    coco_stats = get_coco_metrics(result_file, annotation_file)
    LOGGER.info('AP @[IOU=0.5]: %s' % coco_stats)
