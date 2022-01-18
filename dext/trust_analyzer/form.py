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
        model_questions = random.sample(all_questions[:100],
                                        num_model_questions)
        mov_questions = random.sample(all_questions[100:127],
                                      num_mov_questions)
        general_questions = all_questions[(-1 * num_personal_questions):]
        all_questions = model_questions + mov_questions + general_questions
        return all_questions

    def go_to_next_question():
        st.session_state.curr_question_idx += 1
        if (st.session_state.curr_question_idx ==
                st.session_state.total_questions):
            st.session_state.curr_question_idx = 'DONE'

    def store_and_go(result, qno):
        st.session_state.answers[qno] = result
        answer = {qno: result}
        if write_to_firebase:
            result = st.session_state.firebase_table.post(
                '/dexttrustanalysis-default-rtdb/single_answer:', answer)
        go_to_next_question()

    def go_to_next_question_personal():
        st.session_state.curr_question_idx += 6
        if (st.session_state.curr_question_idx ==
                st.session_state.total_questions):
            st.session_state.curr_question_idx = 'DONE'

    def store_personal(result, qno):
        st.session_state.answers[qno] = result
        answer = {qno: result}
        if write_to_firebase:
            result = st.session_state.firebase_table.post(
                '/dexttrustanalysis-default-rtdb/single_answer:', answer)

    def store_final_user_stats(answers):
        if write_to_firebase:
            result = st.session_state.firebase_table.post(
                '/dexttrustanalysis-default-rtdb/all_answer:', answers)

    def display_question(questions, qno):
        question = questions[qno]
        # st.write("Question ID: ", question["unique_id"])
        caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif"> Question %s </div><br>'
        st.markdown(caption % str(st.session_state.curr_question_idx + 1), unsafe_allow_html=True)

        if question["type"] == "one":
            caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif"> ' \
                      '<a href="https://github.com/DeepanChakravarthiPadmanabhan/trust_analysis/blob/main/TaskDescription.pdf" target="_blank"> View detailed TASK I description</a> </div><br>'
            st.markdown(caption, unsafe_allow_html=True)
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
            caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif">' \
                      'The explanations for the detected object is provided by ' \
                      '<b>highlighting the pixels important </b>for the decision-making process. ' \
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
                image_two = Image.open(images[2])
                st.image(image_two)
            result = st.radio(question["question"], question["options"])
            st.button('Next question', on_click=store_and_go,
                      args=(result, question["unique_id"]))

        if question["type"] == "two":
            caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif"> ' \
                      '<a href="https://github.com/DeepanChakravarthiPadmanabhan/trust_analysis/blob/main/TaskDescription.pdf" target="_blank"> View detailed TASK I description</a> </div><br>'
            st.markdown(caption, unsafe_allow_html=True)
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
            caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif">' \
                      'The explanations for the detected object is provided by ' \
                      '<b>highlighting the pixels important </b>for the decision-making process. ' \
                      'The colorbar on the right of the image indicates the pixel importance scale. ' \
                      'Each image explain a particular bounding box coordinate decision.</div>'
            st.markdown(caption, unsafe_allow_html=True)
            st.header('Bounding box decision explanation')
            col1, col2 = st.columns(2)
            with col1:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> Robot A explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_one = Image.open(images[1])
                st.image(image_one)
            with col2:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> Robot B explanation </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_two = Image.open(images[2])
                st.image(image_two)
            result = st.radio(question["question"], question["options"])
            st.button('Next question', on_click=store_and_go,
                      args=(result, question["unique_id"]))

        if question['type'] == "three" or question['type'] == 'four':
            caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif"> ' \
                      '<a href="https://github.com/DeepanChakravarthiPadmanabhan/trust_analysis/blob/main/TaskDescription.pdf" target="_blank"> View detailed TASK II description</a> </div><br>'
            st.markdown(caption, unsafe_allow_html=True)
            images = question["images"]
            caption = '<div style="text-align: left; color: Black; font-size: 20pxx; font-family:sans-serif"> ' \
                      'The images below include all detections (rectangular' \
                      ' box) predicted by an artificial intelligence agent' \
                      ' in a single image and a visual representation of ' \
                      'the explanations corresponding to the detection ' \
                      'in the same color. The explanations are the important' \
                      ' pixels responsible for the decision-making process. ' \
                      'In each image below the explanations are represented ' \
                      'using either dotted pixels, pixels inside elliptical ' \
                      'region or pixels inside an irregular polygon. </div>'
            st.markdown(caption, unsafe_allow_html=True)
            result = st.radio(question["question"], question["options"])

            col1, col2 = st.columns(2)
            with col1:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> Method 1 </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_five = Image.open(images[0])
                st.image(image_five)
            with col2:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> Method 2 </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_six = Image.open(images[1])
                st.image(image_six)

            col1, col2 = st.columns(2)
            with col1:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> Method 3 </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_five = Image.open(images[2])
                st.image(image_five)
            with col2:
                caption = '<div style="text-align: center; color: Black; font-size: 20pxx; font-family:sans-serif"> Method 4 </div>'
                st.markdown(caption, unsafe_allow_html=True)
                image_six = Image.open(images[3])
                st.image(image_six)

            st.button('Next question', on_click=store_and_go,
                      args=(result, question["unique_id"]))

        if question['type'] == 'P1':
            with st.form('Form'):
                age = st.text_input(questions[-6]['question'], '')
                occcupation = st.text_input(questions[-5]['question'], '')
                cs = st.radio(questions[-4]['question'],
                              questions[-4]['options'])
                xai = st.text_input(questions[-3]['question'], '')
                sufficiency = st.radio(questions[-2]['question'],
                                       questions[-2]['options'])
                agreement = st.radio(questions[-1]['question'],
                                     questions[-1]['options'])
                submit = st.form_submit_button('Submit answers')
                if submit:
                    results = {'age': age, 'occupation': occcupation,
                               'cs': cs, 'xai': xai,
                               'sufficiency': sufficiency,
                               'agreement': agreement}
                    store_personal(results, 'personal')
            st.button('Finish', on_click=go_to_next_question_personal)

    num_model_questions = 10
    num_mov_questions = 5
    num_personal_questions = 6
    num_questions_to_show = (num_model_questions + num_mov_questions +
                             num_personal_questions)
    write_to_firebase = True
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
    st.sidebar.text('Terms in detection')
    st.sidebar.image('trust_analysis/coordinates.png')
    st.sidebar.text('Pixel importance colorbar (heatmap)')
    st.sidebar.image('trust_analysis/heatmap2.png')
    st.sidebar.text('Email: depa03@dfki.de')
    st.sidebar.text('Mobile: +49-15145738603')
    st.sidebar.markdown('<h5>Created by Deepan Chakravarthi Padmanabhan</h5>',
                        unsafe_allow_html=True)

    if st.session_state.curr_question_idx != 'DONE':
        display_question(questions, st.session_state.curr_question_idx)
    else:
        st.success('All questions are answered. '
                   'Thank you for your valuable time.')
        store_final_user_stats(st.session_state.answers)

    if not write_to_firebase:
        st.write('Answers: ')
        st.write(st.session_state.answers)


if __name__ == '__main__':
    show()
