import streamlit as st
from PIL import Image
import numpy as np
from dext.visualizer.interact import get_detections, get_saliency, interactions
from dext.visualizer.interact import interactions_real
from dext.postprocessing.saliency_visualization import convert_to_fig
from io import BytesIO


def load_image(image_file):
    img = Image.open(image_file)
    img = np.array(img)
    return img


def run_interactions():
    values = interactions(image_file, detector_chosen,
                          st.session_state.object_index_chosen,
                          decision_chosen,
                          box_offset_mapping_idx[box_offset_chosen],
                          st.session_state.curr_percent,
                          st.session_state.saliency,
                          st.session_state.box_indices, image_size)
    changed_det_fig = values[0]
    xmin, ymin, xmax, ymax = values[1], values[2], values[3], values[4]
    confidence = values[5]
    modified_fig = values[6]
    box = str(xmin) + ', ' + str(ymin) + ', ' + str(xmax) + ', ' + str(ymax)
    st.session_state.box_values = f'<div style="color: Gray; font-size: 20pxx; font-family:sans-serif"> Bounding box="{box}"</div>'
    conf = str(confidence)
    st.session_state.conf_values = f'<div style="color: Gray; font-size: 20pxx; font-family:sans-serif"> Confidence="{conf}"</div>'
    st.session_state.changed_det_fig = changed_det_fig
    st.session_state.modified_fig = modified_fig

    all_det_on_modified_fig = interactions_real(
        image_file, detector_chosen, st.session_state.object_index_chosen,
        decision_chosen, box_offset_mapping_idx[box_offset_chosen],
        st.session_state.curr_percent, st.session_state.saliency,
        st.session_state.box_indices, image_size)
    st.session_state.all_det_on_modified_fig = all_det_on_modified_fig


@st.cache(suppress_st_warning=True)
def initializations():
    st.session_state.old_object_index = None
    st.session_state.old_decision = 'Classification'
    st.session_state.old_bb = 'None'
    st.session_state.curr_image = 'available'


def run_detections():
    detections, box_indices, det_fig = get_detections(
        image_file, detector_chosen, image_size)
    st.session_state.detections = detections
    st.session_state.box_indices = box_indices
    st.session_state.det_fig = det_fig


def run_saliencies():
    values = get_saliency(image_file, detector_chosen, method_chosen,
                          st.session_state.object_index_chosen,
                          st.session_state.detections,
                          st.session_state.box_indices, decision_chosen,
                          box_offset_mapping_idx[box_offset_chosen])
    single_det_fig, sal_fig = values[0], values[1]
    st.session_state.single_det_fig = single_det_fig
    st.session_state.sal_fig = sal_fig
    saliency = values[2]
    st.session_state.saliency = saliency


apptitle = 'DExT: Analyze Object Detector'
st.set_page_config(page_title=apptitle, page_icon="🔍", layout="wide")

title = "Detector Explanation Toolkit: Interactive Analysis of Detectors"
st.title(title)

image_size = 512
box_offset_mapping_idx = {'x_min': 0, 'y_min': 1, 'x_max': 2, 'y_max': 3,
                          'None': 'None'}
detector_chosen = st.sidebar.selectbox(
    'Detector: ', ['SSD512', 'EFFICIENTDETD0', 'FasterRCNN'])
method_chosen = st.sidebar.selectbox(
    'Select explanation method to interpret: ',
    ['GuidedBackpropagation', 'IntegratedGradients',
     'SmoothGrad_IntegratedGradients', 'SmoothGrad_GuidedBackpropagation'])
decision_chosen = st.sidebar.selectbox(
            'Select decision to explain: ', ['Classification', 'Bounding Box'])
if decision_chosen == 'Bounding Box':
    box_offset_choices = ['x_min', 'y_min', 'x_max', 'y_max']
else:
    box_offset_choices = ['None']
box_offset_chosen = st.sidebar.selectbox(
    'Select bounding box coordinate: ', box_offset_choices)

image_file = st.file_uploader("Upload image to analyze",
                                  type=["png", "jpg", "jpeg"])
if image_file:
    print('Using uploaded file')
    image = load_image(image_file)
    st.session_state.image_file = image
    curr_image = 'uploaded'
else:
    print('Using available file')
    image_file = 'data/000000162701.jpg'
    image = load_image(image_file)
    st.session_state.image_file = image
    curr_image = 'available'

initializations()
if 'old_detector' not in st.session_state:
    run_detections()
    st.session_state.old_detector = detector_chosen
if (st.session_state.old_detector != detector_chosen
        or st.session_state.curr_image != curr_image):
    run_detections()
if (len(st.session_state.detections) > 0
        and st.session_state.old_object_index is None):
    st.session_state.old_object_index = 1000  # large index

drop_down = [i + 1 for i in range(len(st.session_state.detections))]
st.session_state.object_index_chosen = st.sidebar.selectbox(
    'Select object to analyze', drop_down)

if (st.session_state.old_object_index != st.session_state.object_index_chosen
        or st.session_state.old_decision != decision_chosen
        or st.session_state.old_bb != box_offset_chosen
        or st.session_state.curr_image != curr_image
        or st.session_state.old_detector != detector_chosen):
    print('Chosen idx: ', st.session_state.object_index_chosen)
    run_saliencies()
    st.session_state.old_object_index = st.session_state.object_index_chosen
    st.session_state.old_decision = decision_chosen
    st.session_state.old_bb = box_offset_chosen
    st.session_state.curr_image = curr_image
    st.session_state.old_detector = detector_chosen

col1, col2 = st.columns(2)
with col1:
    st.subheader('Input Image')
    st.pyplot(convert_to_fig(st.session_state.image_file))
with col2:
    st.subheader('Detections by %s' % detector_chosen)
    st.pyplot(st.session_state.det_fig)

col1, col2 = st.columns(2)
with col1:
    st.subheader('Chosen Detection')
    buf = BytesIO()
    st.session_state.single_det_fig.savefig(buf, format='png')
    st.image(buf, use_column_width=True)
with col2:
    st.subheader('Explanation - Important Pixels')
    buf = BytesIO()
    st.session_state.sal_fig.savefig(buf, format='png')
    st.image(buf, use_column_width=True)


st.session_state.old_percent = 0
percent = st.sidebar.slider(
    'Percentage of most important pixels to change: ', 0.0, 1.0, 0.8)
st.session_state.curr_percent = percent

run_interactions()

col1, col2 = st.columns(2)
with col1:
    st.subheader('Primal Box Change')
    buf = BytesIO()
    st.session_state.changed_det_fig.savefig(buf, format='png')
    st.image(buf, use_column_width=True)
    st.markdown(st.session_state.box_values, unsafe_allow_html=True)
    st.markdown(st.session_state.conf_values, unsafe_allow_html=True)
with col2:
    st.subheader('Realistic Detection Change')
    buf = BytesIO()
    st.session_state.all_det_on_modified_fig.savefig(buf, format='png')
    st.image(buf, use_column_width=True)


col1, col2 = st.columns(2)
with col1:
    st.subheader('Manipulated Image')
    buf = BytesIO()
    st.session_state.modified_fig.savefig(buf, format='png')
    st.image(buf, use_column_width=True)
with col2:
    st.subheader('Notes')
    st.markdown('Primal Box Details')
    st.markdown(st.session_state.box_values, unsafe_allow_html=True)
    st.markdown(st.session_state.conf_values, unsafe_allow_html=True)
