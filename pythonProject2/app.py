import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np
import hashlib
import matplotlib

matplotlib.use('Agg')
import pickle
from db import *
from PIL import Image
import keras
import cv2 
from matplotlib.image import imread
import os


import sqlite3
conn = sqlite3.connect('usersdata.db',check_same_thread=False)
c = conn.cursor()

# Functions

# def create_usertable():
# 	c.execute('CREATE TABLE IF NOT EXISTS userstable(username TEXT,password TEXT)')
#
#
# def add_userdata(username,password):
# 	c.execute('INSERT INTO userstable(username,password) VALUES (?,?)',(username,password))
# 	conn.commit()
#
# def login_user(username,password):
# 	c.execute('SELECT * FROM userstable WHERE username =? AND password = ?',(username,password))
# 	data = c.fetchall()
# 	return data
#
#
#
# def view_all_users():
# 	c.execute('SELECT * FROM userstable')
# 	data = c.fetchall()
# 	return data

# Password
def generate_hashes(password):
    return hashlib.sha256(str.encode(password)).hexdigest()


def verify_hashes(password, hashed_text):
    if generate_hashes(password) == hashed_text:
        return hashed_text
    return False





# Load ML Models


def load_model(model_file):
    loaded_model = pickle.load(open(model_file, 'rb'))
    return loaded_model


# ML Interpretation
html_temp = """
		<div style="background-color:{};padding:10px;border-radius:10px">
		<h1 style="color:white;text-align:center;">Disease Mortality Prediction </h1>
		<h5 style="color:white;text-align:center;">Hepatitis B </h5>
		</div>
		"""

# Avatar Image using a url
avatar1 = "https://www.w3schools.com/howto/img_avatar1.png"
avatar2 = "https://www.w3schools.com/howto/img_avatar2.png"

result_temp = """
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/img_avatar.png" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

result_temp2 = """
	<div style="background-color:#464e5f;padding:10px;border-radius:10px;margin:10px;">
	<h4 style="color:white;text-align:center;">Algorithm:: {}</h4>
	<img src="https://www.w3schools.com/howto/{}" alt="Avatar" style="vertical-align: middle;float:left;width: 50px;height: 50px;border-radius: 50%;" >
	<br/>
	<br/>	
	<p style="text-align:justify;color:white">{} % probalibilty that Patient {}s</p>
	</div>
	"""

prescriptive_message_temp = """
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Recommended Life style modification</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Get Plenty of Rest</li>
		<li style="text-align:justify;color:black;padding:10px">Exercise Daily</li>
		<li style="text-align:justify;color:black;padding:10px">Avoid Alcohol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Management</h3>
		<ul>
		<li style="text-align:justify;color:black;padding:10px">Consult your doctor</li>
		<li style="text-align:justify;color:black;padding:10px">Take your interferons</li>
		<li style="text-align:justify;color:black;padding:10px">Go for checkups</li>
		<ul>
	</div>
	"""

descriptive_message_temp = """
	<div style="background-color:silver;overflow-x: auto; padding:10px;border-radius:5px;margin:10px;">
		<h3 style="text-align:justify;color:black;padding:10px">Definition</h3>
		<p>Information about the system</p>
	</div>
	"""

STYLE="""
<style>
img{
max-width:100%;
}
</style>

"""

@st.cache



def change_avatar(sex):
    if sex == "male":
        avatar_img = 'img_avatar.png'
    else:
        avatar_img = 'img_avatar2.png'
    return avatar_img


def main():
    """Hep Mortality Prediction App"""
    # st.title("Hepatitis Mortality Prediction App")
    st.markdown(html_temp.format('royalblue'), unsafe_allow_html=True)

    menu = ["Home", "Login", "SignUp"]
    sub_menu = ["Heart", "Diabetes","Pneumonia","Diabetic Retinopathy"]

    choice = st.sidebar.selectbox("Menu", menu)
    if choice == "Home":
        st.subheader("Home")
        # st.text("What is Hepatitis?")
        st.markdown(descriptive_message_temp, unsafe_allow_html=True)

    elif choice == "Login":
        username = st.sidebar.text_input("Username")
        password = st.sidebar.text_input("Password", type='password')
        if st.sidebar.checkbox("Login"):
            create_usertable()
            hashed_pswd = generate_hashes(password)
            result = login_user(username, verify_hashes(password, hashed_pswd))

            if result:
                st.success("Welcome {}".format(username))

                activity = st.selectbox("Checkup", sub_menu)

                if activity == "Heart":
                    st.subheader("Predictive Analytics")

                    sex = st.selectbox("Gender", (0, 1))
                    age = st.slider('Age', 21, 81, 39)
                    currentSmoker = st.selectbox("Do you consume cigarettes?", (0, 1))
                    cigsPerDay=st.slider('How many Cigerattes do you consume per day?', 0, 50, 3)
                    diabetes = st.selectbox("Do you have diabetes?", (0, 1))
                    BPMeds=st.selectbox("Are you on BP Medicines?", (0, 1))
                    prevalentStroke = st.selectbox("Have you experienced a stroke?", (0, 1))
                    prevalentHyp = st.selectbox("Have you experienced Hypertension?", (0, 1))
                    totChol = st.slider('Cholestrol', 0.0, 846.0, 195.0)
                    sysBP = st.slider('Systolic Blood Pressure', 0.0, 67.1, 106.0)
                    diaBP = st.slider('Diabolic Blood Pressure', 0, 500, 77)
                    BMI=st.slider("Body Mass Index(BMI)",0.0,23.41,50.0)
                    heartrate= st.slider("Heart Rate", 50, 150, 80)
                    glucose = st.slider("Glucose", 0, 150, 50)



                    # Store data in dictionary
                    user_data = {
                        'male': sex,
                        'age': age,
                        'currentSmoker': currentSmoker,
                        'cigsPerDay': cigsPerDay,
                        'BPMeds': BPMeds,
                        'prevalentStroke': prevalentStroke,
                        'prevalentHyp': prevalentHyp,
                        'diabetes': diabetes,
                        'totChol': totChol,
                        'sysBP': sysBP,
                        'diaBP': diaBP,
                        'BMI': BMI,
                        'heartRate': heartrate,
                        'glucose': glucose

                    }
                    user_input = pd.DataFrame(user_data, index=[0])
                    st.subheader("User Input")
                    st.write(user_input)

                    if st.button("Predict"):
                        model=load_model('heart.sav')
                        prediction = model.predict(user_input)
                        # Display Predictions
                        st.subheader('Result')
                        st.write(prediction)
                        pred_prob = model.predict_proba(user_input)



                        if prediction == 1:
                            st.warning("Heart Disease Detected")
                            pred_probability_score = {"Heart Disease present": pred_prob[0][0] * 100, "Heart Disease not present": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)
                            st.subheader("Prescriptive Analytics")
                            st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

                        else:
                            st.success("Everything looks fine")
                            pred_probability_score = {"Heart Disease present": pred_prob[0][0] * 100, "Heart Disease not present": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)

                if activity=="Diabetes":
                    st.subheader("Predictive Analytics")

                    pregnancies = st.slider('pregnancies', 0, 17, 1)
                    glucose = st.slider('glucose', 0, 199, 85)
                    blood_pressure = st.slider('blood_pressure', 0, 122, 66)
                    skin_thickness = st.slider('skin_thickness', 0, 99, 29)
                    insulin = st.slider('insulin', 0, 846, 0)
                    BMI = st.slider('BMI', 0.0, 67.1, 26.6)
                    DPF = st.slider('DPF', 0.078, 2.42, 0.351)
                    age = st.slider('age', 21, 81, 31)

                    user_data = {'pregnancies': pregnancies,
                                 'glucose': glucose,
                                 'blood_pressure': blood_pressure,
                                 'skin_thickness': skin_thickness,
                                 'insulin': insulin,
                                 'BMI': BMI,
                                 'DPF': DPF,
                                 'age': age
                                 }
                    # Transform data to Dataframe
                    user_input = pd.DataFrame(user_data, index=[0])
                    st.subheader("User Input")
                    st.write(user_input)

                    if st.button("Predict"):
                        model = load_model('diabetes.sav')
                        prediction = model.predict(user_input)
                        # Display Predictions
                        st.subheader('Result')
                        st.write(prediction)
                        pred_prob = model.predict_proba(user_input)

                        # st.write(prediction)

                        if prediction == 1:
                            st.warning("Patient Has Diabetes")
                            pred_probability_score = {"Diabetic": pred_prob[0][0] * 100, "Non-Diabetic": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)
                            st.subheader("Prescriptive Analytics")
                            st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

                        else:
                            st.success("Everything looks fine")
                            pred_probability_score = {"Diabetic": pred_prob[0][0] * 100, "Non-Diabetic": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)

                if activity=="Pneumonia":
                    st.subheader("Predictive Analytics...")
                    uploaded_file=st.file_uploader("Upload Your File",type="jpeg")
                    if uploaded_file is not None:
                        image=Image.open(uploaded_file)
                        st.image(image, caption='Uploaded Image.', use_column_width=True)
                        st.write("")

                        test_data = []
                        testimg = image.resize((150,150), Image.NEAREST)
                        testimg = np.dstack([testimg, testimg, testimg])
                        testimg = testimg.astype('float32') / 255
                        test_data.append(testimg)
                        user_input= np.array(test_data)

                        if st.button("Predict"):
                            model = keras.models.load_model('lungs1.pkl')
                            prediction = model.predict(user_input)
                            prediction=np.round(prediction)
                            # Display Predictions
                            st.subheader('Result')
                            st.write(prediction)
                            pred_prob = model.predict_proba(user_input)
                            if prediction == 1:
                                st.warning("Patient Has Pneumonia")
                                # pred_probability_score = {"Diabetic": pred_prob[0][0] * 100, "Non-Diabetic": pred_prob[0][1] * 100}
                                # st.subheader("Prediction Probability Score")
                                # st.json(pred_probability_score)
                                st.subheader("Prescriptive Analytics")
                                st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

                            else:
                                st.success("Everything looks fine")
                                pred_probability_score = {"Diabetic": pred_prob[0][0] * 100, "Non-Diabetic": pred_prob[0][1] * 100}
                                st.subheader("Prediction Probability Score")
                                st.json(pred_probability_score)




                if activity=="Diabetic Retinopathy":
                    st.subheader("Predictive Analytics...")
                    def file_selector(folder_path=r'.\test'):
                        filenames = os.listdir(folder_path)
                        selected_filename = st.selectbox('Select a file', filenames)
                        return os.path.join(folder_path, selected_filename)

                    filename = file_selector()
                    st.write('You selected `%s`' % filename)
                    image=Image.open(filename)
                    st.image(image, caption='Uploaded Image.', use_column_width=True)      
                    # if uploaded_file is not None:
                    #     image=Image.open(uploaded_file)
                    #     st.image(image, caption='Uploaded Image.', use_column_width=True)
                    #     st.write("")

                        

                        # img = random.choice(os.listdir("test/DR/"))
                        
                        # im = uploaded_file.name
                        # im = imread(uploaded_file)
                        # # im = im.resize((224,224))
                        # # im = np.array(im)
                        # im = im.reshape([1,224,224,3])

                    input_shape = [1,224,224,3]
                    im = imread(filename)
                    im = im.reshape(input_shape)
                                            

                    if st.button("Predict"):
                        model = keras.models.load_model('DR.sav')
                        prediction = model.predict(im)
                        prediction=np.round(prediction[0][0])
                        # Display Predictions
                        st.subheader('Result')
                        st.write(prediction)
                        pred_prob = model.predict_proba(im)
                        if prediction == 1:
                                st.warning("Patient Has Diabetes")
                                pred_probability_score = {"Diabetic": pred_prob[0][0] * 100, "Non-Diabetic": pred_prob[0][1] * 100}
                                st.subheader("Prediction Probability Score")
                                st.json(pred_probability_score)
                                st.subheader("Prescriptive Analytics")
                                st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

                        else:
                            st.success("Everything looks fine")
                            pred_probability_score = {"Diabetic": pred_prob[0][0] * 100, "Non-Diabetic": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)




            else:
                st.warning("Incorrect Username/Password")


    elif choice == "SignUp":
        new_username = st.text_input("User name")
        new_password = st.text_input("Password", type='password')

        confirm_password = st.text_input("Confirm Password", type='password')
        if new_password == confirm_password:
            st.success("Password Confirmed")
        else:
            st.warning("Passwords not the same")

        if st.button("Submit"):
            create_usertable()
            hashed_new_password = generate_hashes(new_password)
            add_userdata(new_username, hashed_new_password)
            st.success("You have successfully created a new account")
            st.info("Login to Get Started")


if __name__ == '__main__':
    main()
