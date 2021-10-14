import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Concatenate
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D

from dext.model.mask_rcnn.utils import get_resnet_features, build_rpn_model
from dext.model.mask_rcnn.utils import generate_pyramid_anchors, norm_boxes
from dext.model.mask_rcnn.utils import fpn_classifier_graph
from dext.model.mask_rcnn.utils import compute_backbone_shapes
from dext.model.mask_rcnn.layers import DetectionLayer, ProposalLayer


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


def mask_rcnn_detection(config, image_size):
    input_image = tf.keras.layers.Input(shape=image_size,
                                        name='input_image')
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
