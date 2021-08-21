# import numpy as np
# import tensorflow as tf
#
# from paz.backend.image.opencv_image import write_image
# from paz.backend.image import resize_image
#
# from dext.model.efficientdet.efficientdet import EFFICIENTDETD0
# from dext.model.efficientdet.utils import raw_images, efficientdet_preprocess
# from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
# from dext.method.integrated_gradient import IntegratedGradients
#
#
# # TODO: Working on this function.
# def efficientdet_ig_explainer():
#
#     model = EFFICIENTDETD0()
#     image_size = model.image_size
#     input_image, image_scales = efficientdet_preprocess(raw_images, image_size)
#     resized_raw_image = resize_image(raw_images, (image_size, image_size))
#
#     # Functional API calling only provides access to intermediate tensors
#     original_dim = (image_size, image_size, 3)
#     original_inputs = tf.keras.Input(shape=(original_dim), name="input")
#     branch_tensors = model.backbone(original_inputs, False, True)
#     feature_levels = branch_tensors[model.min_level:model.max_level + 1]
#     # Build additional input features that are not from backbone.
#     for resample_layer in model.resample_layers:
#         feature_levels.append(resample_layer(
#             feature_levels[-1], False, None))
#     # BiFPN layers
#     fpn_features = model.fpn_cells(feature_levels, False)
#     # Classification head
#     class_outputs = model.class_net(fpn_features, False)
#     # Box regression head
#     box_outputs = model.box_net(fpn_features, False)
#     efdt = tf.keras.Model(inputs=original_inputs, outputs=[class_outputs, box_outputs])
#     class_outputs, box_outputs = efdt(input_image)
#     efdt.summary()
#
#     # Subclassing model does not give access to tensors
#     # outputs = model(original_inputs)
#     # class_outputs, box_outputs = outputs
#
#     image, detections, class_map_idx = efficientdet_postprocess(
#         model, class_outputs, box_outputs, image_scales, raw_images)
#     print(detections)
#     write_image('images/results/paz_postprocess.jpg', image)
#     print('task completed')
#     print('To match class idx: ')
#     print(class_map_idx)
#
#     baseline = np.zeros(shape=(1, model.image_size, model.image_size, raw_images.shape[-1]))
#     m_steps = 10
#     ig = IntegratedGradients(efdt, baseline, layer_name='box_net')
#     ig_attributions = ig.integrated_gradients(
#         image=resized_raw_image, m_steps=m_steps, batch_size=1)
#     ig.plot_attributions(resized_raw_image, ig_attributions, 'a.jpg')
#
# efficientdet_ig_explainer()

import numpy as np
a = np.ones((1, 2, 2, 9))
a[0, 1, :, :] = 2
a[0, 0, 0, :] = [1, 2, 3, 4, 1, 2, 3, 4, 9]
a[0, 1, 0, :] = [5, 6, 7, 8, 5, 6, 7, 8, 9]
print("raw value: ", a)
# print("num elements: ", a.shape[0] * a.shape[1] * a.shape[2] * a.shape[3])
b = a.reshape((-1, 3))
print(b)
ge = np.unravel_index(np.ravel_multi_index((2, 3), b.shape), a.shape)
print(ge)

### COCO PLAY

# import cv2
# from dext.dataset.coco_dataset import COCOGenerator
# from dext.model.efficientdet.efficientdet  import EFFICIENTDETD7
# from dext.model.efficientdet.utils import preprocess_images, save_file
# from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess
#
# dataset = COCOGenerator('/media/deepan/externaldrive1/datasets_project_repos/mscoco', 'train2017')
#
# for i, data in enumerate(dataset):
#     image, annotations = data
#     image = image
#     labels = annotations[0]['labels']
#     bboxes = annotations[0]['bboxes']
#
#     model = EFFICIENTDETD7()
#     image_size = (image.shape[0], model.image_size,
#                   model.image_size, image.shape[-1])
#     input_image, image_scales = preprocess_images(image, image_size)
#     outputs = model(input_image)
#     image, detections, class_map_idx = efficientdet_postprocess(
#         model, outputs, image_scales, image)
#     print(detections)
#     save_file('paz_postprocess'+str(i)+'.jpg', image, False)
#     print('task completed')

    # for l, b in zip(labels, bboxes):
    #     image = cv2.rectangle(image.astype('uint8'),
    #                   (int(b[0]), int(b[1])),
    #                   (int(b[2]), int(b[3])),
    #                   (0, 0, 255),
    #                   2)
    #     image = cv2.putText(image.astype('uint8'),
    #                 dataset.coco_label_to_name(dataset.label_to_coco_label(l)),
    #                 (int(b[0]), int(b[1]) - 10),
    #                 cv2.FONT_HERSHEY_SIMPLEX,
    #                 0.5,
    #                 (36, 255, 12),
    #                 2)
    #     cv2.imwrite('mock'+str(i)+'.jpg', image)
