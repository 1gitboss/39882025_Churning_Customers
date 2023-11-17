# app.py
import streamlit as st
import pandas as pd
from keras.models import load_model
from sklearn.preprocessing import StandardScaler
from keras.models import model_from_json
import pickle




model = load_model('model2.h5')

# Function to preprocess input data
def preprocess_input_data(data):
    
    # split types of input
    num_data= data.select_dtypes(include=('int64','float64'))
    cat_data= data.select_dtypes(exclude=('int64','float64'))
      
    
# manually  converting "OnlineSecurity" categorical value to numerical
    OnlineSecurity_mapping = {'No': 0, 'Yes': 1}
    cat_data['OnlineSecurity'] = cat_data['OnlineSecurity'].map(OnlineSecurity_mapping)

    # Convert "Contract" categorical value to numerical
    Contract_mapping = {'month-to-month': 0, 'One year': 1, 'Two year': 2}
    cat_data['Contract'] = cat_data['Contract'].map(Contract_mapping)
    
    
    # Convert "Paperless Billing" categorical value to numerical
    PaperlessBilling_mapping = {'Yes': 0, 'No': 1}
    cat_data['PaperlessBilling'] = cat_data['PaperlessBilling'].map(PaperlessBilling_mapping)

    # Convert "PaymentMethod" categorical value to numerical
    PaymentMethod_mapping = {
        'Electronic check': 0, 'Mailed check': 1, 'Bank transfer (automatic)': 2, 'Credit card (automatic)': 3}
    cat_data['PaymentMethod'] = cat_data['PaymentMethod'].map(PaymentMethod_mapping)


    
    # Load the scaler using pickle
    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)
    

    x_scaled= scaler.transform(num_data)
    
    num_data=pd.DataFrame(x_scaled,columns=num_data.columns)
    
    
    preprocessed_data= pd.concat([num_data,cat_data],axis=1)
   
    return preprocessed_data

# Streamlit app
st.title('Churn Prediction App By John Anatsui')

# User input form
tenure = st.slider('Tenure', min_value=0, max_value=75, value=0)
online_security = st.selectbox('Online Security', ['Yes', 'No'])
contract = st.selectbox('Contract', ['Month-to-month', 'One year', 'Two year'])
paperless_billing = st.selectbox('Paperless Billing', ['Yes', 'No'])
payment_method = st.selectbox('Payment Method', ['Electronic check', 'Mailed check', 'Bank transfer (automatic)', 'Credit card (automatic)'])
monthly_charges = st.number_input('Monthly Charges', value=0.0)
total_charges = st.number_input('Total Charges', value=0.0)

# Create a dictionary with user input
user_data = {
    'tenure': tenure,
    'OnlineSecurity': online_security,
    'Contract': contract,
    'PaperlessBilling': paperless_billing,
    'PaymentMethod': payment_method,
    'MonthlyCharges': monthly_charges,
    'TotalCharges': total_charges
}

# Convert the user input to a DataFrame
user_df = pd.DataFrame([user_data])

col1, col2 = st.columns(spec=[.9, .1])


with col1:
    if st.button("Predict"):
       print('user_df:')
       print(user_df)
       preprocessed_user_data = preprocess_input_data(user_df)
       
       prediction_num = model.predict(preprocessed_user_data)
       print('predict:')
       print(prediction_num)
       if prediction_num>=0.5:
           prediction= "The coustomer would churn"
       else:
           prediction= "The customer would not churn"
       
       
       confidence_factor = (1-(prediction_num[0][0]))*100
        # Display the prediction and confidence factor
       st.write(f"Churn Prediction: {prediction}")
       st.write(f"Confidence Factor: {confidence_factor}")
       
with col2:
    if st.button("Clear"):
        st.session_state = {}

