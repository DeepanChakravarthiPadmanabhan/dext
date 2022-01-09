import os
import numpy as np
import random
import json
import pickle
import matplotlib.pyplot as plt


def get_history_file(results_dir, filename):
    if os.path.exists(results_dir):
        file = os.path.join(results_dir, filename)
        if os.path.exists(file):
            return file
        else:
            raise ValueError('%s file not found' % file)
    else:
        raise ValueError('Results directory not found.')


def make_detection(det_path):
    det_image = plt.imread(det_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(det_image)
    ax.axis('off')
    fig_title = 'Detection'
    plt.text(0.5, 1.08, fig_title, horizontalalignment='center',
             fontsize=14, transform=ax.transAxes)
    # plt.show()
    return fig


def make_class_explanation(class_path, title):
    det_image = plt.imread(class_path)
    fig, ax = plt.subplots(figsize=(6.4, 4.8))
    ax.imshow(det_image)
    ax.axis('off')
    fig_title = ('%s explaining classification decision' % title)
    plt.text(0.5, 1.08, fig_title, horizontalalignment='center',
             fontsize=14, transform=ax.transAxes)
    # plt.show()
    return fig


def make_box_explanation(xmin_path, ymin_path, xmax_path, ymax_path, title):
    fig, axes = plt.subplots(1, 4, figsize=(25, 6))
    ax = axes.flatten()

    xmin_image = plt.imread(xmin_path)
    ax[0].imshow(xmin_image)
    ax[0].axis('off')
    ax[0].set_title('x_min decision')

    ymin_image = plt.imread(ymin_path)
    ax[1].imshow(ymin_image)
    ax[1].axis('off')
    ax[1].set_title('y_min decision')

    xmax_image = plt.imread(xmax_path)
    ax[2].imshow(xmax_image)
    ax[2].axis('off')
    ax[2].set_title('x_max decision')

    ymax_image = plt.imread(ymax_path)
    ax[3].imshow(ymax_image)
    ax[3].axis('off')
    ax[3].set_title('y_max decision')

    plt.subplots_adjust(left=0.03, bottom=0.129, right=0.97, top=0.88,
                        wspace=0.017, hspace=0.11)
    fig_title = '%s explaining bounding box decision' % title
    plt.text(1, 1.1, fig_title, horizontalalignment='center',
             fontsize=14, transform=ax[1].transAxes)
    # plt.show()
    return fig


def get_classification_question(detection_label):
    class_question = ('Robot A and Robot B are explaining classification '
                      'decision of the %s detection shown in the image. '
                      'According to you, Which robot\'s explanation is better'
                      ' understandable to explain the decision?'
                      % detection_label)
    return class_question


def get_box_question(detection_label):
    box_question = ('Robot A and Robot B are explaining bounding box decision'
                    ' of the %s detection shown in the image. According to '
                    'you, Which robot\'s explanation is better understandable '
                    'to explain the decision?' % detection_label)
    return box_question


def generate_model_questions(num_question=30):
    # Fix model explanation questions
    num_models = num_question

    # Select model and method
    selected_models = [random.randint(0, 2) for _ in range(num_models)]
    selected_methods = [random.sample(range(0, 3), 2)
                         for _ in selected_models]

    # Get text data
    file = get_history_file(results_folder, 'saliency_image_paths')
    data = [json.loads(line) for line in open(file, 'r')]
    data = np.array(data)

    # Select image id
    unique_image_ids = np.unique(data[:, 0])
    selected_image_ids = random.choices(unique_image_ids,
                                        k=len(selected_methods))

    # Select detection id
    selected_det_ids = []
    selected_detection_labels = []
    for n, i in enumerate(selected_models):
        model_name = model_names[i]
        id_data = data[(data[:, 0] == selected_image_ids[n]) &
                       (data[:, -2] == model_name)]
        num_dets_in_id = np.unique(id_data[:, 1]).astype('int32')
        rand_det_id = random.randint(0, max(num_dets_in_id))
        selected_det_ids.append(rand_det_id)
        label = data[(data[:, -2] == model_name) &
                     (data[:, 0] == selected_image_ids[n]) &
                     (data[:, 1].astype('int32') == rand_det_id)][0, 4]
        selected_detection_labels.append(label)

    print("Models: ", selected_models)
    print("Interpretation methods: ", selected_methods)
    print("Image ids: ", selected_image_ids)
    print("Detections: ", selected_det_ids)
    print("Detection labels: ", selected_detection_labels)

    # Generate 24 model related questions
    model_question_jsons = []
    question_num = 0
    for n in range(len(selected_models)):
        model_name = model_names[selected_models[n]]

        method1 = interpretation_methods[selected_methods[n][0]]
        method2 = interpretation_methods[selected_methods[n][1]]

        path1 = os.path.join(results_folder, model_name + '_' + method1)
        path2 = os.path.join(results_folder, model_name + '_' + method2)

        image_id = selected_image_ids[n]
        det_id = selected_det_ids[n]

        det_image = ('detid_' + str(det_id) + '_imageid_' + str(image_id)
                     + '.jpg')
        class_sal = ('detid_' + str(det_id) + '_saltype_Classification_None'
                     + '_imageid_' + str(image_id) + '.jpg')
        xmin_sal = ('detid_' + str(det_id) + '_saltype_Boxoffset_None'
                    + '_imageid_' + str(image_id) + '.jpg')
        ymin_sal = ('detid_' + str(det_id) + '_saltype_Boxoffset_1'
                    + '_imageid_' + str(image_id) + '.jpg')
        xmax_sal = ('detid_' + str(det_id) + '_saltype_Boxoffset_2'
                    + '_imageid_' + str(image_id) + '.jpg')
        ymax_sal = ('detid_' + str(det_id) + '_saltype_Boxoffset_3'
                    + '_imageid_' + str(image_id) + '.jpg')

        det_image_path = os.path.join(path1, det_image)
        # det_fig = make_detection(det_image_path)

        class_image_path_1 = os.path.join(path1, class_sal)
        class_image_path_2 = os.path.join(path2, class_sal)
        # class_fig1 = make_class_explanation(class_image_path_1, 'Robot A')
        # class_fig2 = make_class_explanation(class_image_path_2, 'Robot B')

        xmin_image_path_1 = os.path.join(path1, xmin_sal)
        ymin_image_path_1 = os.path.join(path1, ymin_sal)
        xmax_image_path_1 = os.path.join(path1, xmax_sal)
        ymax_image_path_1 = os.path.join(path1, ymax_sal)
        # box_fig1 = make_box_explanation(xmin_image_path_1, ymin_image_path_1,
        #                                 xmax_image_path_1, ymax_image_path_1,
        #                                 'Robot A')

        xmin_image_path_2 = os.path.join(path2, xmin_sal)
        ymin_image_path_2 = os.path.join(path2, ymin_sal)
        xmax_image_path_2 = os.path.join(path2, xmax_sal)
        ymax_image_path_2 = os.path.join(path2, ymax_sal)
        # box_fig2 = make_box_explanation(xmin_image_path_2, ymin_image_path_2,
        #                                 xmax_image_path_2, ymax_image_path_2,
        #                                 'Robot B')

        question = dict()
        question_num += 1
        question['unique_id'] = str(question_num)
        question['type'] = 'one'
        question['question'] = get_classification_question(
            selected_detection_labels[n])
        question['images'] = [
            det_image_path, class_image_path_1, class_image_path_2]
        question['image_captions'] = [
            'Detection', 'Robot A Classification',
            'Robot B Classification']
        question['options'] = responses_model
        model_question_jsons.append(question)

        question = dict()
        question_num += 1
        question['unique_id'] = str(question_num)
        question['type'] = 'two'
        question['question'] = get_box_question(
            selected_detection_labels[n])
        question['images'] = [det_image_path, xmin_image_path_1,
                              ymin_image_path_1, xmax_image_path_1,
                              ymax_image_path_1, xmin_image_path_2,
                              ymin_image_path_2, xmax_image_path_2,
                              ymax_image_path_2]
        question['image_captions'] = [
            'Detection', 'Robot A xmin', 'Robot A ymin', 'Robot A xmax',
            'Robot A ymax', 'Robot B xmin', 'Robot B ymin', 'Robot B xmax',
            'Robot B ymax']
        question['options'] = responses_model
        model_question_jsons.append(question)

    return model_question_jsons


interpretation_methods = ['IntegratedGradients', 'GuidedBackpropagation',
                          'SmoothGrad_IntegratedGradients',
                          'SmoothGrad_GuidedBackpropagation']
model_names = ['EFFICIENTDETD0', 'SSD512', 'FasterRCNN']
responses_model = ["Robot A explanation is \"much better\"",
                   "Robot A explanation is \"slightly better\"",
                   "Both explanations are \"same\"",
                   "Robot A explanation is \"slightly worse\"",
                   "Robot A explanation is \"much worse\""]
responses_visualization = ['1', '2', '3', '4']
results_folder = 'trust_analysis'
all_questions = generate_model_questions()
with open('questions.pkl', 'wb') as f:
    pickle.dump(all_questions, f)
