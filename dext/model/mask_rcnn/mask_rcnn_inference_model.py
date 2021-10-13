from tensorflow.keras.layers import Lambda
from tensorflow.keras.models import Model

from dext.model.mask_rcnn.utils import fpn_classifier_graph
from dext.model.mask_rcnn.utils import build_fpn_mask_graph
from dext.model.mask_rcnn.layers import ProposalLayer, DetectionLayer


def inference(model, config, include_mask):
    keras_model = model.keras_model
    input_image = keras_model.input
    anchors = model.anchors
    feature_maps = keras_model.output

    rpn_class_logits, rpn_class, rpn_bbox = model.RPN(feature_maps)

    rpn_rois = ProposalLayer(proposal_count=config.POST_NMS_ROIS_INFERENCE,
                             nms_threshold=config.RPN_NMS_THRESHOLD,
                             name='ROI',
                             config=config)([rpn_class, rpn_bbox, anchors])

    _, classes, mrcnn_bbox = fpn_classifier_graph(
        rpn_rois, feature_maps[:-1], config=config, train_bn=config.TRAIN_BN)

    detections = DetectionLayer(config, name='mrcnn_detection')(
        [rpn_rois, classes, mrcnn_bbox])

    detection_boxes = Lambda(lambda x: x[..., :4])(detections)

    if include_mask:
        mrcnn_mask = build_fpn_mask_graph(detection_boxes, feature_maps[:-1],
                                          config, train_bn=config.TRAIN_BN)
        inference_model = Model(input_image,
                                [detections, classes, mrcnn_bbox, mrcnn_mask,
                                 rpn_rois, rpn_class, rpn_bbox],
                                name='mask_rcnn')
    else:
        inference_model = Model(input_image, detections, name='mask_rcnn')
    return inference_model