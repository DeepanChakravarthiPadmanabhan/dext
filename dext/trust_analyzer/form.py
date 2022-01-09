import streamlit as st
import pickle
import random
from PIL import Image
from streamlit.report_thread import get_report_ctx
from streamlit.server.server import Server
from firebase import firebase


def show():
    st.set_page_config(page_title='User trust study', layout='wide')
    st.title('Detector Explanation Toolkit: Human-Centric Evaluation')

    def _get_session():
        session_id = get_report_ctx().session_id
        session_info = Server.get_current()._get_session_info(session_id)
        if session_info is None:
            raise RuntimeError("Couldn't get your Streamlit Session object.")
        return session_info.session, session_id

    @st.cache
    def create_firebase():
        firebase_table = firebase.FirebaseApplication("https://dexttrustanalysis-default-rtdb.europe-west1.firebasedatabase.app/", None)
        return firebase_table

    @st.cache
    def get_questions(session_id):
        question_seed = session_id
        random.seed(question_seed)
        with open('questions.pkl', 'rb') as f:
            all_questions = pickle.load(f)
        questions = random.sample(all_questions, 3)
        return questions

    def go_to_next_question():
        st.session_state.curr_question_idx += 1
        if (st.session_state.curr_question_idx ==
                st.session_state.total_questions):
            st.session_state.curr_question_idx = 'DONE'

    def store_and_go(result, qno):
        st.session_state.answers[qno] = result
        answer = {qno: result}
        result = st.session_state.firebase_table.post(
            '/dexttrustanalysis-default-rtdb/single_answer:', answer)
        go_to_next_question()

    def store_final_user_stats(answers):
        result = st.session_state.firebase_table.post(
            '/dexttrustanalysis-default-rtdb/all_answer:', answers)

    def display_question(questions, qno):
        question = questions[qno]
        st.write("Question ID: ", question["unique_id"])
        caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif"> ' \
                  'An object detected by an artificial intelligence system is shown below. ' \
                  'Robot A and Robot B are two robots trying to explain the detection result. ' \
                  '<b>Which Robot\'s explanation is reasonable to the detected object?</div>'
        st.markdown(caption, unsafe_allow_html=True)
        images = question["images"]
        det_image = Image.open(images[0])
        st.header('Detected object')
        col1, col2, col3 = st.columns([2.5, 5, 2.5])
        with col2:
            st.image(det_image)


        if question["type"] == "one":
            caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif">' \
                      'The explanations for the detected object is provided by ' \
                      '<b>highlighting the pixels important </b>for the decision making process. ' \
                      'The colorbar on the right of the image indicates the pixel importance scale. </div>'
            st.markdown(caption, unsafe_allow_html=True)
            st.header("Classification decision explanation")
            col1, col2 = st.columns(2)
            with col1:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> Robot A explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_one = Image.open(images[1])
                st.image(image_one)
            with col2:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> Robot B explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_two = Image.open(images[1])
                st.image(image_two)

        if question["type"] == "two":
            caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif">' \
                      'The explanations for the detected object is provided by ' \
                      '<b>highlighting the pixels important </b>for the decision making process. ' \
                      'The colorbar on the right of the image indicates the pixel importance scale. ' \
                      'Each image explain a particular bounding box coordinate decision.</div>'
            st.markdown(caption, unsafe_allow_html=True)

            st.header('Robot A bounding box decision explanation')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> x_min explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_one = Image.open(images[1])
                st.image(image_one)
            with col2:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> y_min explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_two = Image.open(images[2])
                st.image(image_two)
            with col3:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> x_max explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_three = Image.open(images[3])
                st.image(image_three)
            with col4:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif">  y_max explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_four = Image.open(images[4])
                st.image(image_four)

            st.header('Robot B bounding box decision explanation')
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> x_min explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_five = Image.open(images[5])
                st.image(image_five)
            with col2:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> y_min explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_six = Image.open(images[6])
                st.image(image_six)
            with col3:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> x_max explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_seven = Image.open(images[7])
                st.image(image_seven)
            with col4:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif">  y_max explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_eight = Image.open(images[8])
                st.image(image_eight)

        result = st.radio(question["question"],  question["options"])
        st.button('Next question', on_click=store_and_go,
                  args=(result, question["unique_id"]))

    user_session, session_id = _get_session()
    questions = get_questions(session_id)
    st.session_state.get(questions)

    if "firebase_table" not in st.session_state:
        st.session_state.firebase_table = create_firebase()

    if "questions" not in st.session_state:
        st.session_state.questions = questions
        st.session_state.answers = {}
        st.session_state.curr_question_idx = 0
        st.session_state.total_questions = len(st.session_state.questions)

    st.sidebar.text('Task hints:')
    st.sidebar.text('Pixel importance colorbar (heatmap)')
    st.sidebar.image('trust_analysis/heatmap2.png')
    st.sidebar.text('Email: depa03@dfki.de')
    st.sidebar.text('Mobile: +49-15145738603')
    st.sidebar.markdown('<h5>Created by Deepan Chakravarthi Padmanabhan</h5>',
                        unsafe_allow_html=True)

    if st.session_state.curr_question_idx != 'DONE':
        display_question(questions, st.session_state.curr_question_idx)
    else:
        st.success('All questions answered. Thank you for your valuable time.')
        store_final_user_stats(st.session_state.answers)

    # st.write('Answers: ')
    # st.write(st.session_state.answers)


if __name__ == '__main__':
    show()
