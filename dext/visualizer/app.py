import streamlit as st
from PIL import Image
import numpy as np
from dext.visualizer.interact import get_detections, get_saliency, interactions
import matplotlib.pyplot as plt


def load_image(image_file):
    img = Image.open(image_file)
    img = np.array(img)
    return img


def run_detections():
    print('Running inference')
    detections, box_indices, det_img = get_detections(
        st.session_state.image_file, st.session_state.select_detector,
        image_size)
    st.session_state.detections = detections
    st.session_state.box_indices = box_indices
    st.subheader('Detections made by %s' % st.session_state.select_detector)
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(det_img)


def run_saliency():
    print('Running saliency')
    values = get_saliency(
        st.session_state.image_file, st.session_state.select_detector,
        st.session_state.select_method, st.session_state.select_detection,
        st.session_state.detections, st.session_state.box_indices,
        st.session_state.select_explanation,
        st.session_state.select_box_offset)
    det_fig, sal_fig = values[0], values[1]
    st.session_state.saliency = values[2]
    col1, col2 = st.columns(2)
    with col1:
        st.subheader('Chosen Detection')
        st.pyplot(det_fig, clear_figure=True)
    with col2:
        st.subheader('Explanation')
        st.pyplot(sal_fig, clear_figure=True)


def run_interaction():
    print('Re-running interaction')
    values = interactions(
        st.session_state.image_file, st.session_state.select_detector,
        st.session_state.select_detection, st.session_state.select_explanation,
        st.session_state.select_box_offset, st.session_state.percentage_change,
        st.session_state.saliency, st.session_state.box_indices,
        image_size)
    changed_det_fig = values[0]
    xmin, ymin, xmax, ymax = values[1], values[2], values[3], values[4]
    confidence = values[5]
    modified_fig = values[6]
    col1, col2 = st.columns(2)
    with col1:
        st.pyplot(changed_det_fig, clear_figure=True)
    with col2:
        box = str(xmin) + ', ' + str(ymin) + ', ' + str(xmax) + ', ' + str(
            ymax)
        box_values = f'<div style="color: Black; font-size: 20pxx; font-family:sans-serif"> Bounding box="{box}"</div>'
        conf = str(confidence)
        conf_values = f'<div style="color: Black; font-size: 20pxx; font-family:sans-serif"> Confidence="{conf}"</div>'
        st.markdown(box_values, unsafe_allow_html=True)
        st.markdown(conf_values, unsafe_allow_html=True)
        st.pyplot(modified_fig, clear_figure=True)


def set_explanation_params(select_method, select_explanation,
                           select_box_offset, select_detection):
    st.session_state.select_method = select_method
    st.session_state.select_explanation = select_explanation
    st.session_state.select_detection = select_detection

    if st.session_state.select_explanation == 'Classification':
        st.session_state.select_box_offset = None
        st.write('Setting box offset to None as decision chosen is Classification')
    else:
        st.session_state.select_box_offset = select_box_offset
    return True


def show_form():
    with st.form('Form-Exp'):
        select_method = st.selectbox(
            'Select explanation method to interpret: ',
            [None, 'IntegratedGradients', 'GuidedBackpropagation',
             'SmoothGrad_IntegratedGradients',
             'SmoothGrad_GuidedBackpropagation'])
        select_explanation = st.selectbox(
            'Select decision to explain: ',
            [None, 'Classification', 'Bounding Box'])
        select_box_offset = st.selectbox(
            'Select bounding box coordinate: ',
            ['NA', 'x_min', 'y_min', 'x_max', 'y_max'])
        drop_down = [i + 1 for i in range(len(st.session_state.detections))]
        select_detection = st.selectbox(
            'Select object to analyze', drop_down)
        submit = st.form_submit_button('Get explanation')
        if submit:
            set_status = set_explanation_params(
                select_method, select_explanation, select_box_offset,
                select_detection)
            if set_status:
                st.session_state.run_saliency = True
                st.session_state.got_detector = False
                st.session_state.got_image = False


def run_interaction_only():
    st.session_state.got_image = False
    st.session_state.got_detector = False
    st.session_state.run_saliency = False
    st.session_state.run_saliency = False
    run_interaction()


@st.cache
def initialize_session_state():
    st.session_state.got_image = False
    st.session_state.got_detector = False
    st.session_state.run_saliency = False
    st.session_state.start_interaction = False


apptitle = 'DExT: Analyze Object Detector'
st.set_page_config(page_title=apptitle, page_icon="🔍", layout="wide")

title = "Detector Explanation Toolkit: Interactive Analysis of Detectors"
st.title(title)

image_size = 512
initialize_session_state()

st.session_state.image_file = st.file_uploader("Upload image to analyze",
                                               type=["png", "jpg", "jpeg"])
if st.session_state.image_file:
    st.session_state.got_image =True

if st.session_state.got_image:
    st.subheader('Selected input image to interpret')
    st.image(load_image(st.session_state.image_file))
    image = load_image(st.session_state.image_file)
    with st.form('Form-Det'):
        select_detector = st.selectbox(
            'Select detector: ',
            [None, 'SSD512', 'EFFICIENTDETD0', 'FasterRCNN'])
        submit = st.form_submit_button('Get detector')
        if submit:
            st.session_state.select_detector = select_detector
            st.session_state.got_detector = True
            st.session_state.got_image = False

if st.session_state.got_detector:
    run_detections()
    show_form()

if st.session_state.run_saliency:
    run_saliency()
    st.session_state.start_interaction = True

if st.session_state.start_interaction:
    st.session_state.percentage_change = st.slider(
        'Percentage of most important pixels to change: ', 0.0, 1.0, 0.8,
    on_change=run_interaction_only)
    run_interaction()
