import numpy as np

from paz.processors.image import LoadImage
from paz.abstract import SequentialProcessor

from dext.model.mask_rcnn.mask_rcnn_preprocess import ResizeImages
from dext.model.mask_rcnn.mask_rcnn_preprocess import  NormalizeImages
from dext.model.mask_rcnn.model import MaskRCNN
from dext.model.mask_rcnn.inference_graph import InferenceGraph
from dext.model.mask_rcnn.config import Config
from dext.model.mask_rcnn.utils import norm_boxes_graph
from dext.model.mask_rcnn.mask_rcnn_postprocess import mask_rcnn_postprocess
from dext.model.mask_rcnn.utils import compute_backbone_shapes, norm_boxes
from dext.model.mask_rcnn.utils import generate_pyramid_anchors
from dext.interpretation_method.integrated_gradient import \
    IntegratedGradientExplainer


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80

def get_anchors(image_shape, config):
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    anchor_cache = {}
    if not tuple(image_shape) in anchor_cache:
        anchors = generate_pyramid_anchors(
            config.RPN_ANCHOR_SCALES,
            config.RPN_ANCHOR_RATIOS,
            backbone_shapes,
            config.BACKBONE_STRIDES,
            config.RPN_ANCHOR_STRIDE)
        anchors = anchors
        anchor_cache[tuple(image_shape)] = norm_boxes(
            anchors, image_shape[:2])
    return anchor_cache[tuple(image_shape)]


def test(image, weights_path):
    config = TestConfig()
    resize = SequentialProcessor([ResizeImages(config)])
    molded_images, windows = resize(image)
    image_shape = molded_images[0].shape
    window = norm_boxes_graph(windows[0], image_shape[:2])
    config.WINDOW = window

    base_model = MaskRCNN(config=config, model_dir=weights_path)
    inference_model = InferenceGraph(model=base_model, config=config,
                                     include_mask=False)
    base_model.keras_model = inference_model()
    base_model.keras_model.load_weights(weights_path, by_name=True)
    model = base_model.keras_model

    preprocess = SequentialProcessor([ResizeImages(config),
                                      NormalizeImages(config)])
    normalized_images, windows = preprocess(image)
    image_shape = normalized_images[0].shape
    anchors = get_anchors(image_shape, config)
    anchors = np.broadcast_to(anchors, (config.BATCH_SIZE,) + anchors.shape)

    detections = model.predict([normalized_images, anchors])
    results = mask_rcnn_postprocess(image[0], normalized_images[0], windows[0],
                                    detections[0])
    # TODO: Do IG or GBP and see the results after getting interest IDX
    return results

raw_image = "images/000000128224.jpg"
weights_path = '/media/deepan/externaldrive1/project_repos/DEXT_versions/weights/mask_rcnn_coco.h5'

loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
results = test([image], weights_path)
print(results)
print("done")



