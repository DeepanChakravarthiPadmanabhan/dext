# import logging
# import matplotlib.pyplot as plt
# import tensorflow as tf
# from tensorflow.keras.models import Model
# import innvestigate
#
# from paz.backend.image import resize_image
# from dext.model.functional_models import get_functional_model
# from dext.postprocessing.saliency_visualization import \
#     visualize_saliency_grayscale
#
# LOGGER = logging.getLogger(__name__)
#
#
# class OD_LRP:
#     def __init__(self, model, model_name, image, explainer,
#                  layer_name=None, visualize_idx=None,
#                  preprocessor_fn=None, image_size=512):
#         self.model = model
#         self.model_name = model_name
#         self.image = image
#         self.explainer = explainer
#         self.layer_name = layer_name
#         self.visualize_idx = visualize_idx
#         self.preprocessor_fn = preprocessor_fn
#         self.image_size = image_size
#         self.image = self.check_image_size(self.image, self.image_size)
#         self.image = self.preprocess_image(self.image, self.image_size)
#         if self.layer_name is None:
#             self.layer_name = self.find_target_layer()
#         self.custom_model = self.build_custom_model()
#
#     def check_image_size(self, image, image_size):
#         if image.shape != (image_size, image_size, 3):
#             image = resize_image(image, (image_size, image_size))
#         return image
#
#     def find_target_layer(self):
#         for layer in reversed(self.model.layers):
#             if len(layer.output_shape) == 4:
#                 return layer.name
#         raise ValueError(
#             "Could not find 4D layer. Cannot apply guided backpropagation.")
#
#     def preprocess_image(self, image, image_size):
#         preprocessed_image = self.preprocessor_fn(image, image_size)
#         if type(preprocessed_image) == tuple:
#             input_image, image_scales = preprocessed_image
#         else:
#             input_image = preprocessed_image
#         return input_image
#
#     def build_custom_model(self):
#         if "EFFICIENTDET" in self.model_name:
#             self.model = get_functional_model(
#                 self.model_name, self.model)
#         else:
#             self.model = self.model
#
#         print(self.visualize_idx, "INDEX SEEING")
#         custom_model = Model(inputs=[self.model.inputs],
#                              outputs=[tf.expand_dims(tf.expand_dims(
#                                  self.model.output[self.visualize_idx[0],
#                                                    self.visualize_idx[1],
#                                                    self.visualize_idx[2]],
#                                  axis=0), axis=0)]
#                              # outputs=[self.model.output]
#                              )
#         if 'SSD' in self.model_name:
#             for i in custom_model.layers:
#                 if hasattr(i, 'activation'):
#                     if i.activation == tf.keras.activations.softmax:
#                         i.activation = tf.keras.activations.linear
#         return custom_model
#
#     def get_saliency_map(self):
#         inputs = tf.cast(self.image, tf.float32)
#         # out = self.custom_model(inputs)
#         # print("Model out shape: ", out.shape)
#         # print('Out row: ', out[0, :5, 0])
#         # print("Out raw shape: ", out.shape)
#         # sel_idx = [
#         # (out.shape[2] * self.visualize_idx[1]) + self.visualize_idx[2]]
#         # print('selected idx: ', sel_idx)
#         # print('Out selected: ', out[0, self.visualize_idx[1],
#         # self.visualize_idx[2]])
#         analyzer = innvestigate.create_analyzer('lrp.z', self.custom_model)
#         a = analyzer.analyze(inputs,
#                              # neuron_selection= sel_idx #2018435
#                              )
#         print(a.keys())
#         saliency = a['image']
#         return saliency
#
#
# def OD_LRPExplainer_WIP(model, model_name, image, interpretation_method,
#                         layer_name, visualize_index, preprocessor_fn,
#                         image_size):
#     explainer = OD_LRP(model, model_name, image, interpretation_method,
#                        layer_name, visualize_index, preprocessor_fn,
#                        image_size)
#     saliency = explainer.get_saliency_map()
#     saliency = visualize_saliency_grayscale(saliency)
#     plt.imsave('saliency.jpg', saliency)
#     return saliency
#
#
# def OD_LRPExplainer(model, model_name, image, interpretation_method,
#                     layer_name, visualize_index, preprocessor_fn, image_size):
#     pass
