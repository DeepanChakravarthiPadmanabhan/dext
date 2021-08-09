import cv2
from dext.dataset.coco_dataset import COCOGenerator
from dext.model.efficientdet.efficientdet  import EFFICIENTDETD7
from dext.model.efficientdet.utils import preprocess_images, save_file
from dext.model.efficientdet.efficientdet_postprocess import efficientdet_postprocess

dataset = COCOGenerator('/media/deepan/externaldrive1/datasets_project_repos/mscoco', 'train2017')

for i, data in enumerate(dataset):
    image, annotations = data
    image = image
    labels = annotations[0]['labels']
    bboxes = annotations[0]['bboxes']

    model = EFFICIENTDETD7()
    image_size = (image.shape[0], model.image_size,
                  model.image_size, image.shape[-1])
    input_image, image_scales = preprocess_images(image, image_size)
    outputs = model(input_image)
    image, detections, class_map_idx = efficientdet_postprocess(
        model, outputs, image_scales, image)
    print(detections)
    save_file('paz_postprocess'+str(i)+'.jpg', image, False)
    print('task completed')

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
