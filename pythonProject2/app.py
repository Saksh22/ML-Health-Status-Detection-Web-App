import streamlit as st

# EDA Pkgs
import pandas as pd
import hashlib

import matplotlib

matplotlib.use('Agg')
import pickle
from db import *



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
    sub_menu = ["Heart", "Diabetes"]

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
                    trestbps = st.slider('Resting blood pressure', 0, 50, 0)
                    chol = st.slider('Cholestrol', 0.0, 846.0, 195.0)
                    fbs = st.selectbox('Fasting blood sugar', (0,1))
                    restecg = st.slider('Resting electrocardiographic results', 0.0, 67.1, 106.0)
                    thalach = st.slider('Maximum heart rate achieved', 0, 500, 77)
                    exang = st.selectbox("Exercise induced angina", (0, 1))
                    oldpeak=st.slider("ST depression induced by exercise relative to rest",0.0,1.4,50.0)
                    ca=st.selectbox("Number of major vessels ", (0, 1))
                    cp_0=st.selectbox("Chest pain type 0 ", (0, 1))
                    cp_1 = st.selectbox("Chest pain type 1 ", (0, 1))
                    cp_2 = st.selectbox("Chest pain type 2 ", (0, 1))
                    cp_3 = st.selectbox("Chest pain type 3 ", (0, 1))
                    thal_0 = st.selectbox(" Defect Type 1 = normal", (0, 1))
                    thal_1 = st.selectbox(" Defect Type 2 = normal", (0, 1))
                    thal_2 = st.selectbox(" Defect Type 3 = normal", (0, 1))
                    thal_3 = st.selectbox(" Defect Type 4 = normal", (0, 1))
                    slope_0 = st.selectbox("The slope of the peak exercise ST segment 1", (0, 1))
                    slope_1 = st.selectbox("The slope of the peak exercise ST segment 2", (0, 1))
                    slope_2 = st.selectbox("The slope of the peak exercise ST segment 3", (0, 1))


                    # Store data in dictionary
                    user_data = {
                        'age': age,
                        'sex': sex,
                        'trestbps': trestbps,
                        'chol': chol,
                        'fbs': fbs,
                        'restecg': restecg,
                        'thalach': thalach,
                        'exang': exang,
                        'oldpeak': oldpeak,
                        'ca': ca,
                        'cp_0': cp_0,
                        'cp_1': cp_1,
                        'cp_2': cp_2,
                        'cp_3': cp_3,
                        'thal_0': thal_0,
                        'thal_1': thal_1,
                        'thal_2': thal_2,
                        'thal_3': thal_3,
                        'slope_0': slope_0,
                        'slope_1': slope_1,
                        'slope_2': slope_2

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
                            pred_probability_score = {"Die": pred_prob[0][0] * 100, "Live": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)
                            st.subheader("Prescriptive Analytics")
                            st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

                        else:
                            st.success("Everything looks fine")
                            pred_probability_score = {"Die": pred_prob[0][0] * 100, "Live": pred_prob[0][1] * 100}
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
                            pred_probability_score = {"Die": pred_prob[0][0] * 100, "Live": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)
                            st.subheader("Prescriptive Analytics")
                            st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

                        else:
                            st.success("Everything looks fine")
                            pred_probability_score = {"Die": pred_prob[0][0] * 100, "Live": pred_prob[0][1] * 100}
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
