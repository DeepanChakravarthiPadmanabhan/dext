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


def get_box_question(detection_label, label):
    box_question = ('Robot A and Robot B are explaining %s bounding box '
                    'decision of the %s detection shown in the image.'
                    ' According to you, Which robot\'s explanation is better'
                    ' understandable to explain the decision?'
                    % (label, detection_label))
    return box_question


def generate_model_questions(num_question=50):
    # Fix model explanation questions
    num_models = num_question

    # Select model and method
    selected_models = [random.randint(0, 2) for _ in range(num_models)]
    selected_methods = [random.sample(range(0, 4), 2)
                         for _ in selected_models]

    # Get text data
    file = get_history_file(model_folder, 'saliency_image_paths')
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

        path1 = os.path.join(model_folder, model_name + '_' + method1)
        path2 = os.path.join(model_folder, model_name + '_' + method2)

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

        a_images = [xmin_image_path_1, ymin_image_path_1, xmax_image_path_1,
                    ymax_image_path_1]
        b_images = [xmin_image_path_2, ymin_image_path_2, xmax_image_path_2,
                    ymax_image_path_2]
        coordinate_choice = random.randint(0, 3)
        label = ['x_left_top', 'y_left_top', 'x_right_bottom',
                 'y_right_bottom']
        a_choice = a_images[coordinate_choice]
        b_choice = b_images[coordinate_choice]
        a_caption = 'Robot A ' + label[coordinate_choice]
        b_caption = 'Robot B ' + label[coordinate_choice]

        question['question'] = get_box_question(
            selected_detection_labels[n], label[coordinate_choice])
        question['images'] = [det_image_path, a_choice, b_choice]
        question['image_captions'] = ['Detection', a_caption, b_caption]
        question['options'] = responses_model
        model_question_jsons.append(question)

    return model_question_jsons


def get_class_mov_questions():
    class_question = ('Below given 4 images provide the detections and the '
                      'image regions more important to categorize the'
                      ' corresponding objects in 4 different ways. Which '
                      'image is better understandable for you?')
    return class_question


def get_box_mov_questions(q_type):
    box_question = ('Below given 4 images provide the detections and the '
                    'image regions more important to predict the %s of the'
                    ' corresponding objects in 4 different ways. Which image '
                    'is better understandable for you?' % q_type)
    return box_question


def get_mov_images(image_id, type):
    file_dir = os.path.join(mov_folder, type)
    ellipse = os.path.join(
        file_dir, image_id + '_EFFICIENTDETD0_' + 'ellipse.jpg')
    contours = os.path.join(
        file_dir, image_id + '_EFFICIENTDETD0_' + 'contours.jpg')
    dbscan_points = os.path.join(
        file_dir, image_id + '_EFFICIENTDETD0_' + 'dbscan_points.jpg')
    convex = os.path.join(
        file_dir, image_id + '_EFFICIENTDETD0_' + 'convex.jpg')
    images = [ellipse, contours, dbscan_points, convex]
    return images


def generate_mov_questions(num_questions, question_num):
    all_image_ids = ['2010_001212', '46400', '50384', '102731', '109741',
                     '152137', '162701', '422994', '575232', '2008_000008',
                     '2008_000144', '2008_000251', '2010_005511',
                     '2011_007177', '2012_000072', '2007_001239',
                     '2007_000847', '2007_001408', '2007_005264',
                     '2007_005273', '2008_000519', '2008_001448',
                     '2008_001852', '2008_002601', '2008_004296',
                     '2008_005929', '2008_006117']

    all_types = ['class', 'class', 'class', 'class', 'class', 'class', 'class',
                 'class', 'class', 'class', 'class', 'class', 'class', 'class',
                 'class', 'class', 'x_left_top', 'x_left_top', 'y_left_top',
                 'y_left_top', 'x_right_bottom', 'x_right_bottom',
                 'x_right_bottom', 'y_right_bottom', 'y_right_bottom',
                 'y_right_bottom', 'y_right_bottom']

    # class_image_ids = ['2010_001212', '46400', '50384', '102731', '109741',
    #                    '152137', '162701', '422994', '575232', '2008_000008',
    #                    '2008_000144', '2008_000251', '2010_005511',
    #                    '2011_007177', '2012_000072', '2007_001239']
    # xmin_image_ids = ['2007_000847', '2007_001408']
    # ymin_image_ids = ['2007_005264', '2007_005273']
    # xmax_image_ids = ['2008_000519', '2008_001448', '2008_001852']
    # ymax_image_ids = ['2008_002601', '2008_004296', '2008_005929',
    #                   '2008_006117']

    # selected_image_ids = random.choices(all_image_ids, k=num_questions)
    selected_image_ids = all_image_ids
    # selected_types = [all_types[all_image_ids.index(i)]
    #                   for n, i in enumerate(selected_image_ids)]
    selected_types = all_types

    model_question_jsons = []
    for n, i in enumerate(selected_image_ids):
        question = dict()
        if selected_types[n] == 'class':
            question['unique_id'] = str(question_num)
            question['type'] = 'three'
            question['question'] = get_class_mov_questions()
            question['images'] = get_mov_images(i, selected_types[n])
            question['options'] = responses_visualization
        else:
            question['unique_id'] = str(question_num)
            question['type'] = 'four'
            question['question'] = get_box_mov_questions(selected_types[n])
            question['images'] = get_mov_images(i, selected_types[n])
            question['options'] = responses_visualization

        question_num += 1
        model_question_jsons.append(question)

    return model_question_jsons


def generate_general_questions(question_num):
    model_question_jsons = []
    question = dict()
    question['unique_id'] = str(question_num)
    question['type'] = 'P1'
    question['question'] = 'Please enter your age below.'
    model_question_jsons.append(question)
    question_num += 1

    question = dict()
    question['unique_id'] = str(question_num)
    question['type'] = 'P2'
    question['question'] = 'Please enter your job title/occupation below.'
    model_question_jsons.append(question)
    question_num += 1

    question = dict()
    question['unique_id'] = str(question_num)
    question['type'] = 'P3'
    question['question'] = 'Are you working in the field of Computer Science?'
    question['options'] = ['Yes', 'No']
    model_question_jsons.append(question)
    question_num += 1

    question = dict()
    question['unique_id'] = str(question_num)
    question['type'] = 'P4'
    question['question'] = 'Have you worked in eXplainable AI? If yes, please provide the keywords related to your work.'
    model_question_jsons.append(question)
    question_num += 1

    question = dict()
    question['unique_id'] = str(question_num)
    question['type'] = 'P5'
    question['question'] = 'Was the task description provided understandable to you and sufficient to perform the task successfully?'
    question['options'] = ['Yes', 'No']
    model_question_jsons.append(question)
    question_num += 1

    question = dict()
    question['unique_id'] = str(question_num)
    question['type'] = 'P6'
    question['question'] = 'Please provide your consent to use your answers for research purpose.'
    question['options'] = ['Yes', 'No']
    model_question_jsons.append(question)
    question_num += 1

    return model_question_jsons


interpretation_methods = ['IntegratedGradients', 'GuidedBackpropagation',
                          'SmoothGrad_IntegratedGradients',
                          'SmoothGrad_GuidedBackpropagation']
model_names = ['EFFICIENTDETD0', 'SSD512', 'FasterRCNN']
model_folder = 'trust_analysis'
responses_model = ["Robot A explanation is \"much better\"",
                   "Robot A explanation is \"slightly better\"",
                   "Both explanations are \"same\"",
                   "Robot A explanation is \"slightly worse\"",
                   "Robot A explanation is \"much worse\""]
responses_visualization = ['Method 1', 'Method 2', 'Method 3', 'Method 4',
                           'None of the methods']

model_questions = generate_model_questions(50)
print('Total model questions: ', len(model_questions))
mov_folder = 'mov'
mov_questions = generate_mov_questions(27, len(model_questions) + 1)
print('Total mov questions: ', len(mov_questions))
general_questions = generate_general_questions((len(mov_questions) +
                                                len(model_questions) + 1))
all_questions = model_questions + mov_questions + general_questions
print('Total questions: ', len(all_questions))
print('All questions: ', all_questions)
with open('questions.pkl', 'wb') as f:
    pickle.dump(all_questions, f)
