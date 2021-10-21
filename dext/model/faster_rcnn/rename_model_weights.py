import h5py
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Concatenate
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D

from dext.model.faster_rcnn.utils import get_resnet_features, build_rpn_model
from paz.processors.image import LoadImage
from dext.model.faster_rcnn.faster_rcnn_preprocess import mask_rcnn_preprocess
from dext.model.faster_rcnn.utils import generate_pyramid_anchors
from dext.model.faster_rcnn.utils import norm_boxes_graph, norm_boxes
from dext.model.faster_rcnn.utils import fpn_classifier_graph
from dext.model.faster_rcnn.utils import compute_backbone_shapes
from dext.model.faster_rcnn.layers import DetectionLayer, ProposalLayer
from dext.model.faster_rcnn.config import Config
from dext.model.faster_rcnn.faster_rcnn_postprocess import mask_rcnn_postprocess


def read_hdf5(path):
    """A function to read weights from h5 file."""
    weights = {}
    keys = []
    with h5py.File(path, 'r') as f:
        f.visit(keys.append)
        for key in keys:
            if ':' in key:
                weights[f[key].name] = f[key][()]
    return weights


def get_anchors(image_shape, config):
    backbone_shapes = compute_backbone_shapes(config, image_shape)
    anchor_cache = {}
    anchors = generate_pyramid_anchors(
        config.RPN_ANCHOR_SCALES,
        config.RPN_ANCHOR_RATIOS,
        backbone_shapes,
        config.BACKBONE_STRIDES,
        config.RPN_ANCHOR_STRIDE)
    anchor_cache[tuple(image_shape)] = norm_boxes(
        anchors, image_shape[:2])
    anchors = anchor_cache[tuple(image_shape)]
    anchors = np.broadcast_to(anchors,
                              (config.BATCH_SIZE,) + anchors.shape)
    return anchors


def rpn_layer(rpn_feature_maps, config):
    rpn = build_rpn_model(config.RPN_ANCHOR_STRIDE,
                          len(config.RPN_ANCHOR_RATIOS),
                          config.TOP_DOWN_PYRAMID_SIZE)
    layer_outputs = [rpn([feature]) for feature in rpn_feature_maps]
    names = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
    outputs = list(zip(*layer_outputs))
    outputs = [Concatenate(axis=1, name=name)(list(output))
               for output, name in zip(outputs, names)]
    return outputs


def maskrcnn_detection_model(config, image_size):
    input_image = tf.keras.layers.Input(shape=image_size, name='input_image')
    if callable(config.BACKBONE):
        _, C2, C3, C4, C5 = config.BACKBONE(
            input_image, stage5=True, train_bn=config.TRAIN_BN)
    else:
        _, C2, C3, C4, C5 = get_resnet_features(
            input_image, config.BACKBONE, True, config.TRAIN_BN)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1), name='fpn_c5p5')(C5)
    upsample_P5 = UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5)
    conv2d_P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1),
                       name='fpn_c4p4')(C4)
    P4 = Add(name='fpn_p4add')([upsample_P5, conv2d_P4])

    upsample_P4 = UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4)
    conv2d_P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1),
                       name='fpn_c3p3')(C3)
    P3 = Add(name='fpn_p3add')([upsample_P4, conv2d_P3])

    upsample_P3 = UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3)
    conv2d_P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (1, 1),
                       name='fpn_c2p2')(C2)
    P2 = Add(name='fpn_p2add')([upsample_P3, conv2d_P2])

    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                padding='SAME', name='fpn_p2')(P2)
    P3 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                padding='SAME', name='fpn_p3')(P3)
    P4 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                padding='SAME', name='fpn_p4')(P4)
    P5 = Conv2D(config.TOP_DOWN_PYRAMID_SIZE, (3, 3),
                padding='SAME', name='fpn_p5')(P5)
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)

    feature_maps = [P2, P3, P4, P5, P6]

    rpn_class_logits, rpn_class, rpn_bbox = rpn_layer(feature_maps, config)
    anchors = get_anchors(image_size, config)
    rpn_rois = ProposalLayer(
        proposal_count=config.POST_NMS_ROIS_INFERENCE,
        nms_threshold=config.RPN_NMS_THRESHOLD,
        name='ROI',
        config=config)([rpn_class, rpn_bbox, anchors])
    _, classes, mrcnn_bbox = fpn_classifier_graph(
        rpn_rois, feature_maps[:-1], config=config,
        train_bn=config.TRAIN_BN)
    detections = DetectionLayer(
        config, name='mrcnn_detection')([rpn_rois, classes, mrcnn_bbox])
    model = tf.keras.models.Model(
        inputs=input_image, outputs=detections, name='mask_rcnn')
    return model


class TestConfig(Config):
    NAME = "test"
    GPU_COUNT = 1
    IMAGES_PER_GPU = 1
    NUM_CLASSES = 1 + 80


def test(image, weights_path):
    config = TestConfig()
    normalized_images, window = mask_rcnn_preprocess(config, image)
    image_size = normalized_images[0].shape

    config_window = norm_boxes_graph(window[0], image_size[:2])
    config.WINDOW = config_window

    model = maskrcnn_detection_model(config=config, image_size=image_size)

    pretrained_weights = read_hdf5(weights_path)
    all_pretrained_weights = list(pretrained_weights.keys())
    print('LENGTH OF PRETRAINED: ', len(all_pretrained_weights))
    print('LENGTH OF MODEL WEIGHT: ', len(model.weights))

    for i in all_pretrained_weights:
        print(i)
    print('Printed all pretrained weights')

    for i in model.weights:
        print(i.name)
    print("Printed model weights")

    success = 0
    different_appenders = ['rpn_conv_shared', 'rpn_class_raw', 'rpn_bbox_pred']
    for n, i in enumerate(model.weights):
        name = i.name
        if any(substring in name for substring in different_appenders):
            appender = '/rpn_model/'
            name = appender + name
        else:
            appender = name.split('/')[0]
            name = '/' + appender + '/' + name
        print("TRYING: NUM: ", n, ": NAME: ", name)
        if name in all_pretrained_weights:
            if model.weights[n].shape == pretrained_weights[name].shape:
                model.weights[n].assign(pretrained_weights[name])
                success = success + 1
                print('copying', success)
            else:
                raise ValueError('Shape mismatch for weights of same name.')
        else:
            print('not copying due to no name', name)
            raise ValueError("Weight with %s not found." % name)

    print('DONE COPYING WEIGHTS')
    model.save_weights('new_weights_maskrcnn.h5')

    # model.load_weights('new_weights_maskrcnn.h5')

    input_image = tf.expand_dims(normalized_images[0], axis=0)
    detections = model(input_image)
    detections = detections.numpy()
    detection, image = mask_rcnn_postprocess(image[0], normalized_images[0],
                                             window[0], detections[0])
    return detection, image


raw_image = "images/000000128224.jpg"
weights_path = 'new_weights_maskrcnn.h5'
loader = LoadImage()
raw_image = loader(raw_image)
image = raw_image.copy()
detection, image = test([image], weights_path)
print(detection)
plt.imsave('paz_maskrcnn.jpg', image)
print("done")
