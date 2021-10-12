from dext.model.mask_rcnn.model import MaskRCNN
from dext.model.mask_rcnn.config import Config
from dext.model.mask_rcnn.utils import norm_boxes_graph
from dext.model.mask_rcnn.inference_graph import InferenceGraph
from dext.model.mask_rcnn.detection import ResizeImages, NormalizeImages
from dext.model.mask_rcnn.detection import Detect, PostprocessInputs
from paz.abstract import SequentialProcessor


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


def test(images, weights_path):
    config = TestConfig()
    resize = SequentialProcessor([ResizeImages(config)])
    molded_images, windows = resize(images)
    image_shape = molded_images[0].shape
    window = norm_boxes_graph(windows[0], image_shape[:2])
    config.WINDOW = window

    base_model = MaskRCNN(config=config, model_dir='../../mask_rcnn')
    inference_model = InferenceGraph(
        model=base_model, config=config, include_mask=True)
    base_model.keras_model = inference_model()
    base_model.keras_model.load_weights(weights_path, by_name=True)
    preprocess = SequentialProcessor([ResizeImages(config),
                                      NormalizeImages(config)])
    postprocess = SequentialProcessor([PostprocessInputs()])
    detect = Detect(base_model, config, preprocess, postprocess)
    results = detect(images)
    return results
