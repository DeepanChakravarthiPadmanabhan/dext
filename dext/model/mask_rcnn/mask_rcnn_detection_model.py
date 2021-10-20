import gin
import numpy as np
import tensorflow as tf
from tensorflow.keras.layers import Add, Conv2D, Concatenate
from tensorflow.keras.layers import UpSampling2D, MaxPooling2D

from dext.model.mask_rcnn.utils import get_resnet_features, build_rpn_model
from dext.model.mask_rcnn.utils import generate_pyramid_anchors, norm_boxes
from dext.model.mask_rcnn.utils import fpn_classifier_graph
from dext.model.mask_rcnn.utils import compute_backbone_shapes
from dext.model.mask_rcnn.layers import DetectionLayer, ProposalLayer
from dext.model.mask_rcnn.mask_rcnn_preprocess import mask_rcnn_preprocess
from dext.model.mask_rcnn.utils import norm_boxes_graph


def get_anchors(image_shape, backbone='resnet101', batch_size=1,
                compute_backbone_shape=None,
                rpn_anchor_scales=(32, 64, 128, 256, 512),
                rpn_anchor_ratios=[0.5, 1, 2],
                backbone_strides=[4, 8, 16, 32, 64],
                rpn_anchor_stride=1):
    backbone_shapes = compute_backbone_shapes(image_shape, backbone,
                                              compute_backbone_shape,
                                              backbone_strides)
    anchor_cache = {}
    anchors = generate_pyramid_anchors(rpn_anchor_scales, rpn_anchor_ratios,
                                       backbone_shapes, backbone_strides,
                                       rpn_anchor_stride)
    anchor_cache[tuple(image_shape)] = norm_boxes(
        anchors, image_shape[:2])
    anchors = anchor_cache[tuple(image_shape)]
    anchors = np.broadcast_to(anchors,
                              (batch_size,) + anchors.shape)
    return anchors


def rpn_layer(rpn_feature_maps, rpn_anchor_stride=1,
              rpn_anchor_ratios=[0.5, 1, 2], top_down_pyramid_size=256):
    rpn = build_rpn_model(rpn_anchor_stride,
                          len(rpn_anchor_ratios),
                          top_down_pyramid_size)
    layer_outputs = [rpn([feature]) for feature in rpn_feature_maps]
    names = ['rpn_class_logits', 'rpn_class', 'rpn_bbox']
    outputs = list(zip(*layer_outputs))
    outputs = [Concatenate(axis=1, name=name)(list(output))
               for output, name in zip(outputs, names)]
    return outputs


def mask_rcnn_detection(image_size, window, train_bn=False,
                        backbone="resnet101", top_down_pyramid_size=256,
                        post_nms_rois_inference=1000, rpn_nms_threshold=0.7,
                        batch_size=1, compute_backbone_shape=None,
                        rpn_anchor_scales=(32, 64, 128, 256, 512),
                        rpn_anchor_ratios=[0.5, 1, 2],
                        backbone_strides=[4, 8, 16, 32, 64],
                        rpn_anchor_stride=1,
                        rpn_bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
                        pre_nms_limit=6000, images_per_gpu=1, pool_size=7,
                        num_classes=81, image_max_dim=1024,
                        bbox_std_dev=np.array([0.1, 0.1, 0.2, 0.2]),
                        detection_max_instances=100,
                        detection_min_confidence=0.7,
                        detection_nms_threshold=0.3):
    input_image = tf.keras.layers.Input(shape=image_size,
                                        name='input_image')

    _, C2, C3, C4, C5 = get_resnet_features(
        input_image, backbone, True, train_bn)
    P5 = Conv2D(top_down_pyramid_size, (1, 1), name='fpn_c5p5')(C5)
    upsample_P5 = UpSampling2D(size=(2, 2), name='fpn_p5upsampled')(P5)
    conv2d_P4 = Conv2D(top_down_pyramid_size, (1, 1),
                       name='fpn_c4p4')(C4)
    P4 = Add(name='fpn_p4add')([upsample_P5, conv2d_P4])

    upsample_P4 = UpSampling2D(size=(2, 2), name='fpn_p4upsampled')(P4)
    conv2d_P3 = Conv2D(top_down_pyramid_size, (1, 1),
                       name='fpn_c3p3')(C3)
    P3 = Add(name='fpn_p3add')([upsample_P4, conv2d_P3])

    upsample_P3 = UpSampling2D(size=(2, 2), name='fpn_p3upsampled')(P3)
    conv2d_P2 = Conv2D(top_down_pyramid_size, (1, 1),
                       name='fpn_c2p2')(C2)
    P2 = Add(name='fpn_p2add')([upsample_P3, conv2d_P2])

    # Attach 3x3 conv to all P layers to get the final feature maps.
    P2 = Conv2D(top_down_pyramid_size, (3, 3),
                padding='SAME', name='fpn_p2')(P2)
    P3 = Conv2D(top_down_pyramid_size, (3, 3),
                padding='SAME', name='fpn_p3')(P3)
    P4 = Conv2D(top_down_pyramid_size, (3, 3),
                padding='SAME', name='fpn_p4')(P4)
    P5 = Conv2D(top_down_pyramid_size, (3, 3),
                padding='SAME', name='fpn_p5')(P5)
    P6 = MaxPooling2D(pool_size=(1, 1), strides=2, name='fpn_p6')(P5)

    feature_maps = [P2, P3, P4, P5, P6]

    rpn_class_logits, rpn_class, rpn_bbox = rpn_layer(
        feature_maps, rpn_anchor_stride, rpn_anchor_ratios,
        top_down_pyramid_size)
    anchors = get_anchors(image_size, backbone, batch_size,
                          compute_backbone_shape, rpn_anchor_scales,
                          rpn_anchor_ratios, backbone_strides,
                          rpn_anchor_stride)
    rpn_rois = ProposalLayer(post_nms_rois_inference, rpn_nms_threshold,
                             rpn_bbox_std_dev, pre_nms_limit, images_per_gpu,
                             name='ROI')([rpn_class, rpn_bbox, anchors])
    _, classes, mrcnn_bbox = fpn_classifier_graph(
        rpn_rois, feature_maps[:-1], pool_size, num_classes, image_max_dim,
        train_bn=train_bn)
    detections = DetectionLayer(
        batch_size, window, bbox_std_dev, images_per_gpu,
        detection_max_instances, detection_min_confidence,
        detection_nms_threshold,
        name='mrcnn_detection')([rpn_rois, classes, mrcnn_bbox])
    model = tf.keras.models.Model(
        inputs=input_image, outputs=detections, name='mask_rcnn')
    return model


@gin.configurable
def get_maskrcnn_model(image, image_size, weight_path):
    normalized_images, image_scales = mask_rcnn_preprocess(image, image_size)
    normalized_image_size = normalized_images[0].shape
    config_window = norm_boxes_graph(image_scales, (image_size, image_size))
    model = mask_rcnn_detection(image_size=normalized_image_size,
                                window=config_window)
    model.load_weights(weight_path + 'mask_rcnn.h5')
    return model
