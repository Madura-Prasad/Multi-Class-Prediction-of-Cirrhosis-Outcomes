import streamlit as st
import pandas as pd
import pickle

# Load the trained RandomForestClassifier model
model_file_path = 'Cirrhosis_Outcomes.pkl'
model = pickle.load(open(model_file_path, 'rb'))

# Define expected feature names used during training
expected_features = ['N_Days', 'Age', 'Bilirubin', 'Cholesterol', 'Albumin', 'Copper', 
                     'Alk_Phos', 'SGOT', 'Tryglicerides', 'Platelets', 'Prothrombin', 
                     'Stage', 'Drug_Placebo', 'Sex_M', 'Ascites_Y', 'Hepatomegaly_Y', 
                     'Spiders_Y', 'Edema_S', 'Edema_Y', 'Edema_N']

# Mapping dictionary for status labels
status_mapping = {0: "D", 1: "C", 2: "CL"}

# Streamlit app
def main():
    # Custom CSS to set background color and apply additional styles
    custom_css = """
    <style>
        .reportview-container {
            background: #ffffff; /* Change the background color */
        }
        
        .title {
            color: #333333; /* Change the title color */
            text-align: center; /* Center align the title */
            font-size: 30px; /* Adjust the font size of the title */
            margin-bottom: 20px; /* Add margin below the title */
        }
    </style>
    """
    st.markdown(custom_css, unsafe_allow_html=True)

    st.title('Multi-Class Prediction of Cirrhosis Outcomes')


    n_days = st.number_input('N Days')
    age = st.number_input('Age')
    bilirubin = st.number_input('Bilirubin')
    cholesterol = st.number_input('Cholesterol')
    albumin = st.number_input('Albumin')
    copper = st.number_input('Copper')
    alk_phos = st.number_input('Alk Phos')
    sgot = st.number_input('SGOT')
    tryglicerides = st.number_input('Tryglicerides')
    platelets = st.number_input('Platelets')
    prothrombin = st.number_input('Prothrombin')
    stage = st.number_input('Stage')
    drug_p = st.number_input('Drug Placebo')
    sex_m = st.number_input('Sex_M')
    ascites_y = st.number_input('Ascites_Y')
    hepatomegaly_y = st.number_input('Hepatomegaly_Y')
    spiders_y = st.number_input('Spiders_Y')
    edema_s = st.number_input('Edema_S')
    edema_y = st.number_input('Edema_Y')
    edema_n = st.number_input('Edema_N')

    # Prepare input data for prediction
    input_data = pd.DataFrame({
        'N_Days': [n_days],
        'Age': [age],
        'Bilirubin': [bilirubin],
        'Cholesterol': [cholesterol],
        'Albumin': [albumin],
        'Copper': [copper],
        'Alk_Phos': [alk_phos],
        'SGOT': [sgot],
        'Tryglicerides': [tryglicerides],
        'Platelets': [platelets],
        'Prothrombin': [prothrombin],
        'Stage': [stage],
        'Drug_Placebo': [drug_p],
        'Sex_M': [sex_m],
        'Ascites_Y': [ascites_y],
        'Hepatomegaly_Y': [hepatomegaly_y],
        'Spiders_Y': [spiders_y],
        'Edema_S': [edema_s],
        'Edema_Y': [edema_y],
        'Edema_N': [edema_n]
    })

    # Make predictions
    if st.button('Predict'):
        # Make predictions using the loaded model
        predictions = model.predict_proba(input_data)

        # Get the predicted class index
        predicted_class_index = predictions.argmax(axis=1)

        # Convert the predicted class index to status label
        predicted_labels = [status_mapping[index] for index in predicted_class_index]

        # Style the message with different colors
        status_color = ['green' if label == "D" else 'darkred' if label == "C" else 'orange' for label in predicted_labels]
        colored_text = [f'<span style="color: {color}; font-size: 20px;">Predicted Status: {label}</span>' for label, color in zip(predicted_labels, status_color)]

        # Display predictions
        st.subheader('Predictions')
        for text in colored_text:
            st.markdown(text, unsafe_allow_html=True)

# Run the app
if __name__ == '__main__':
    main()
