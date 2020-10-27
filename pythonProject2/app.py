import streamlit as st

# EDA Pkgs
import pandas as pd
import numpy as np

# Utils
import os
import joblib
import hashlib
# passlib,bcrypt

# Data Viz Pkgs
import matplotlib.pyplot as plt
import matplotlib

matplotlib.use('Agg')

# DB
from db import *
import pickle

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
import lime
import lime.lime_tabular

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
		<li style="text-align:justify;color:black;padding:10px">Avoid Alchol</li>
		<li style="text-align:justify;color:black;padding:10px">Proper diet</li>
		<ul>
		<h3 style="text-align:justify;color:black;padding:10px">Medical Mgmt</h3>
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
		<p>Hepatitis B is a viral infection that attacks the liver and can cause both acute and chronic disease.</p>
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
            # if password == "12345":
            if result:
                st.success("Welcome {}".format(username))

                activity = st.selectbox("Checkup", sub_menu)
                # if activity == "Plot":
                #     st.subheader("Data Vis Plot")
                #     df = pd.read_csv("data/clean_hepatitis_dataset.csv")
                #     st.dataframe(df)
                #
                #     df['class'].value_counts().plot(kind='bar')
                #     st.pyplot()
                #
                #     # Freq Dist Plot
                #     freq_df = pd.read_csv("data/freq_df_hepatitis_dataset.csv")
                #     st.bar_chart(freq_df['count'])
                #
                #     if st.checkbox("Area Chart"):
                #         all_columns = df.columns.to_list()
                #         feat_choices = st.multiselect("Choose a Feature", all_columns)
                #         new_df = df[feat_choices]
                #         st.area_chart(new_df)



                if activity == "Heart":
                    st.subheader("Predictive Analytics")

                    Sex_male = st.selectbox("Gender", (0, 1))
                    age = st.slider('age', 21, 81, 39)
                    cigsPerDay = st.slider('cigsPerDay', 0, 50, 0)
                    totChol = st.slider('Total Cholestrol', 0.0, 846.0, 195.0)
                    BMI = st.slider('BMI', 0.0, 67.1, 26.97)
                    sysBP = st.slider('sysBP', 0.0, 67.1, 106.0)
                    glucose = st.slider('glucose', 0, 500, 77)

                    # Store data in dictionary
                    user_data = {'Sex_male': Sex_male,
                                 'age': age,
                                 'cigsPerDay': cigsPerDay,
                                 'totChol': totChol,
                                 'sysBP': sysBP,
                                 'BMI': BMI,
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


                        # st.write(prediction)
                        # prediction_label = {"Die":1,"Live":2}
                        # final_result = get_key(prediction,prediction_label)
                        if prediction == 1:
                            st.warning("Patient Dies")
                            pred_probability_score = {"Die": pred_prob[0][0] * 100, "Live": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)
                            st.subheader("Prescriptive Analytics")
                            st.markdown(prescriptive_message_temp, unsafe_allow_html=True)

                        else:
                            st.success("Patient Lives")
                            pred_probability_score = {"Die": pred_prob[0][0] * 100, "Live": pred_prob[0][1] * 100}
                            st.subheader("Prediction Probability Score")
                            st.json(pred_probability_score)

                    # if st.checkbox("Interpret"):
                    #     if model_choice == "KNN":
                    #         loaded_model = load_model("models/knn_hepB_model.pkl")
                    #
                    #     elif model_choice == "DecisionTree":
                    #         loaded_model = load_model("models/decision_tree_clf_hepB_model.pkl")
                    #
                    #     else:
                    #         loaded_model = load_model("models/logistic_regression_hepB_model.pkl")
                    #
                    #         # loaded_model = load_model("models/logistic_regression_model.pkl")
                    #         # 1 Die and 2 Live
                    #         df = pd.read_csv("data/clean_hepatitis_dataset.csv")
                    #         x = df[['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites', 'varices',
                    #                 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime', 'histology']]
                    #         feature_names = ['age', 'sex', 'steroid', 'antivirals', 'fatigue', 'spiders', 'ascites',
                    #                          'varices', 'bilirubin', 'alk_phosphate', 'sgot', 'albumin', 'protime',
                    #                          'histology']
                    #         class_names = ['Die(1)', 'Live(2)']
                    #         explainer = lime.lime_tabular.LimeTabularExplainer(x.values, feature_names=feature_names,
                    #                                                            class_names=class_names,
                    #                                                            discretize_continuous=True)
                    #         # The Explainer Instance
                    #         exp = explainer.explain_instance(np.array(feature_list), loaded_model.predict_proba,
                    #                                          num_features=13, top_labels=1)
                    #         exp.show_in_notebook(show_table=True, show_all=False)
                    #         # exp.save_to_file('lime_oi.html')
                    #         st.write(exp.as_list())
                    #         new_exp = exp.as_list()
                    #         label_limits = [i[0] for i in new_exp]
                    #         # st.write(label_limits)
                    #         label_scores = [i[1] for i in new_exp]
                    #         plt.barh(label_limits, label_scores)
                    #         st.pyplot()
                    #         plt.figure(figsize=(20, 10))
                    #         fig = exp.as_pyplot_figure()
                    #         st.pyplot()
                    #
                    #
                    #
                    #
                    #

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
