import pandas as pd 
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from keras.models import load_model
import streamlit as st
import yfinance as yf
import pickle
import joblib
from sklearn.ensemble import RandomForestClassifier
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import train_test_split
import tensorflow as tf
import random as rn
import yfinance as yf
import datetime as dt
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from sklearn.preprocessing import MinMaxScaler
np.random.seed(1)
tf.random.set_seed(1)
rn.seed(1)
from keras.models import Sequential
from keras.layers import Dense, Dropout, LSTM
from sklearn.metrics import mean_squared_error, mean_absolute_error,r2_score
from millify import millify
import pandas_datareader as web
from cryptocmd import CmcScraper


start = '2010-01-01'
end = '2019-12-31'

st.title('Data Insights')

st.title('Stock Price Prediction')

user_input = st.text_input('Enter Stock Ticker', 'AAPL')

df = yf.download(user_input, start=start, end=end)


# Describing Data
st.subheader('Data from 2010 - 2019')
st.write(df.describe())


# Visualizing Data

st.subheader('Closing Price vs Time Chart')
fig = plt.figure(figsize=(12,6))
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 Moving Average')
ma100 = df.Close.rolling(100).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma100)
plt.plot(df.Close)
st.pyplot(fig)

st.subheader('Closing Price vs Time Chart with 100 and 200 Moving Average')
ma100 = df.Close.rolling(100).mean()
ma200 = df.Close.rolling(200).mean()
fig = plt.figure(figsize=(12,6))
plt.plot(ma200)
plt.plot(df.Close)
st.pyplot(fig)


# Splitting for training and testing

data_training = pd.DataFrame(df['Close'][0:int(len(df) * 0.70)])
data_testing = pd.DataFrame(df['Close'][int(len(df) * 0.70):int(len(df))])

from sklearn.preprocessing import MinMaxScaler
scalar = MinMaxScaler(feature_range=(0,1))

data_training_array = scalar.fit_transform(data_training)

# Loading model

model = load_model('keras_model.h5')

# Testing Part

past_100_days = data_training.tail(100)
final_df = pd.concat([past_100_days, data_testing], ignore_index=True)
input_data = scalar.transform(final_df)

x_test = []
y_test = []

for i in range(100, input_data.shape[0]):
    x_test.append(input_data[i-100 : i])
    y_test.append(input_data[i, 0])

x_test, y_test = np.array(x_test), np.array(y_test)

y_predicted = model.predict(x_test)

scaler = scalar.scale_

scaler = scalar.scale_
scale_factor = 1 / scaler[0]
y_test = y_test * scale_factor
y_predicted = y_predicted * scale_factor

# Final Graph
st.subheader('Predictions vs Original')
fig2 = plt.figure(figsize=(12, 6))
plt.plot(y_test, 'blue', label='Original Price')
plt.plot(y_predicted, 'red', label='Predicted Price')
plt.xlabel('Time')
plt.ylabel('Price')
plt.legend()
st.pyplot(fig2)


# Load the Random Forest model from the pickle file
model = pickle.load(open('randomforest.pkl', 'rb'))

# Define the columns for user input
columns = ['tenure', 'PhoneService', 'Contract',
           'PaperlessBilling', 'PaymentMethod', 'MonthlyCharges']

# Create a function to preprocess user input and make predictions


def predict_churn(input_data):
    # Preprocess the input data
    input_df = pd.DataFrame([input_data], columns=columns)

    # Make predictions using the loaded model
    prediction = model.predict(input_df)
    probability = model.predict_proba(input_df)[:, 1]

    return prediction[0], probability[0]

# Create the Streamlit app


def main():
    st.title("Customer Churn Prediction")
    st.write("Enter the customer details below to predict churn.")

    # Create input fields for user input
    tenure = st.slider("Tenure (months)", 0, 100, 1)
    
    phone_service = st.selectbox("Phone Service", [0, 1])
    st.write("0: No, 1: Yes")
    contract = st.selectbox("Contract", [0, 1, 2])
    st.write("0: Month-to-month, 1: One year, 2: Two year")
    paperless_billing = st.selectbox("Paperless Billing", [0, 1])
    st.write("0: No, 1: Yes")
    payment_method = st.selectbox("Payment Method", [0, 1, 2, 3])
    st.write("0: Bank transfer (automatic), 1: Credit card (automatic), 2: Electronic check, 3: Mailed check")
    monthly_charges = st.number_input("Monthly Charges")

    # Create a dictionary to store the user input
    input_data = {
        'tenure': tenure,
        'PhoneService': phone_service,
        'Contract': contract,
        'PaperlessBilling': paperless_billing,
        'PaymentMethod': payment_method,
        'MonthlyCharges': monthly_charges
    }

    # Predict churn based on user input
    churn_probability = predict_churn(input_data)
    churn_prediction=churn_probability[1]
    # Display the prediction
    st.subheader("Churn Prediction")
    if churn_prediction >= 0.4:
        st.write("The customer is likely to churn.")
    else:
        st.write("The customer is unlikely to churn.")

    # Display the churn probability
    st.subheader("Churn Probability")

    st.write("The probability of churn is:", churn_probability)


# Run the Streamlit app
if __name__ == '__main__':
    main()

st.write("# Credit Card Fraud Detection")

st.sidebar.header('Input Credit Card Details')

uploaded_file = st.sidebar.file_uploader("Upload your input CSV file", type=['csv'])
if uploaded_file is not None:
    input_df = pd.read_csv(uploaded_file)
else:
    def user_input():
        V1 = st.sidebar.slider('V1', -5.0, 1.5, 5.0)
        V2 = st.sidebar.slider('V2', -5.0, 1.5, 5.0)
        V3 = st.sidebar.slider('V3', -5.0, 1.5, 5.0)
        V4 = st.sidebar.slider('V4', -5.0, 1.5, 5.0)
        V5 = st.sidebar.slider('V5', -5.0, 1.5, 5.0)
        V6 = st.sidebar.slider('V6', -5.0, 1.5, 5.0)
        V7 = st.sidebar.slider('V7', -5.0, 1.5, 5.0)
        V8 = st.sidebar.slider('V8', -5.0, 1.5, 5.0)
        V9 = st.sidebar.slider('V9', -5.0, 1.5, 5.0)
        V10 = st.sidebar.slider('V10', -5.0, 1.5, 5.0)
        V11 = st.sidebar.slider('V11', -5.0, 1.5, 5.0)
        V12 = st.sidebar.slider('V12', -5.0, 1.5, 5.0)
        V13 = st.sidebar.slider('V13', -5.0, 1.5, 5.0)
        V14 = st.sidebar.slider('V14', -5.0, 1.5, 5.0)
        V15 = st.sidebar.slider('V15', -5.0, 1.5, 5.0)
        V16 = st.sidebar.slider('V16', -5.0, 1.5, 5.0)
        V17 = st.sidebar.slider('V17', -5.0, 1.5, 5.0)
        V18 = st.sidebar.slider('V18', -5.0, 1.5, 5.0)
        V19 = st.sidebar.slider('V19', -5.0, 1.5, 5.0)
        V20 = st.sidebar.slider('V20', -5.0, 1.5, 5.0)
        V21 = st.sidebar.slider('V21', -5.0, 1.5, 5.0)
        V22 = st.sidebar.slider('V22', -5.0, 1.5, 5.0)
        V23 = st.sidebar.slider('V23', -5.0, 1.5, 5.0)
        V24 = st.sidebar.slider('V24', -5.0, 1.5, 5.0)
        V25 = st.sidebar.slider('V25', -5.0, 1.5, 5.0)
        V26 = st.sidebar.slider('V26', -5.0, 1.5, 5.0)
        V27 = st.sidebar.slider('V27', -5.0, 1.5, 5.0)
        V28 = st.sidebar.slider('V28', -5.0, 1.5, 5.0)
        Amount = st.sidebar.number_input('Amount')

        data = {'V1': V1,
                'V2': V2,
                'V3': V3,
                'V4': V4,
                'V5': V5,
                'V6': V6,
                'V7': V7,
                'V8': V8,
                'V9': V9,
                'V10': V10,
                'V11': V11,
                'V12': V12,
                'V13': V13,
                'V14': V14,
                'V15': V15,
                'V16': V16,
                'V17': V17,
                'V18': V18,
                'V19': V19,
                'V20': V20,
                'V21': V21,
                'V22': V22,
                'V23': V23,
                'V24': V24,
                'V25': V25,
                'V26': V26,
                'V27': V27,
                'V28': V28,
                'Amount': Amount
                }
        fea = pd.DataFrame(data, index=[0])
        return fea


    input_df = user_input()



st.subheader('Credit Card Data')

if uploaded_file is not None:
    st.write(input_df)
else:
    st.write('Awaiting CSV file to be uploaded.')
    st.write(input_df)

load_clf = joblib.load(open('./model.joblib', 'rb'))

prediction = load_clf.predict(input_df)
prediction_probability = load_clf.predict_proba(input_df)

st.subheader('Prediction')
if prediction[0] == 0:
    st.write("Genuine Transaction")
else:
    st.write("Fraudulent Transaction")

st.subheader('Prediction Probability')
st.write(prediction_probability)

st.subheader('Next-Day Forecasting with Long-Short Term Memory (LSTM)')

csv = pd.read_csv('convertcsv.csv')
symbol = csv['symbol'].tolist()

# creating sidebar
ticker_input = st.selectbox('Enter or Choose Crypto Coin', symbol,index=symbol.index('ETH'))

start = dt.datetime.today() - dt.timedelta(5*365)
end = dt.datetime.today()

a = start.strftime('%d-%m-%Y')
b = end.strftime('%d-%m-%Y')

# initialise scraper with time interval for e.g a year from today
scraper = CmcScraper(ticker_input, a, b)
# Pandas dataFrame for the same data
df = scraper.get_dataframe()

st.write('It will take some seconds to fit the model....')
eth_df = df.sort_values(['Date'],ascending=True, axis=0)


#creating dataframe
eth_lstm = pd.DataFrame(index=range(0,len(eth_df)),columns=['Date', 'Close'])
for i in range(0,len(eth_df)):
    eth_lstm['Date'][i] = eth_df['Date'][i]
    eth_lstm['Close'][i] = eth_df['Close'][i]

#setting index
eth_lstm.index = eth_lstm.Date
eth_lstm.drop('Date', axis=1, inplace=True)
eth_lstm = eth_lstm.sort_index(ascending=True)


#creating train and test sets
dataset = eth_lstm.values
train = dataset[0:990,:]
valid = dataset[990:,:]

#converting dataset into x_train and y_train
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(dataset)

x_train, y_train = [], []
for i in range(60,len(train)):
    x_train.append(scaled_data[i-60:i,0])
    y_train.append(scaled_data[i,0])
x_train, y_train = np.array(x_train), np.array(y_train)

x_train = np.reshape(x_train, (x_train.shape[0],x_train.shape[1],1))

# create and fit the LSTM network
model = Sequential()
model.add(LSTM(units=50, return_sequences=True, input_shape=(x_train.shape[1],1)))
model.add(LSTM(units=50))
model.add(Dense(1))

model.compile(loss='mean_squared_error', optimizer='adam')
model.fit(x_train, y_train, epochs=1, batch_size=1, verbose=2)
print('Fitting Model')
#predicting 246 values, using past 60 from the train data
inputs = eth_lstm[len(eth_lstm) - len(valid) - 60:].values
inputs = inputs.reshape(-1,1)
inputs  = scaler.transform(inputs)

X_test = []
for i in range(60,inputs.shape[0]):
    X_test.append(inputs[i-60:i,0])
X_test = np.array(X_test)

X_test = np.reshape(X_test, (X_test.shape[0],X_test.shape[1],1))
closing_price = model.predict(X_test)
closing_price = scaler.inverse_transform(closing_price)

rms=np.sqrt(mean_squared_error(closing_price,valid))
acc = r2_score(closing_price,valid)*100

# for plotting
train = eth_df[:990]
valid = eth_df[990:]
valid['Predictions'] = closing_price

st.write('#### Actual VS Predicted Prices')

fig_preds = go.Figure()
fig_preds.add_trace(
    go.Scatter(
        x=train['Date'],
        y=train['Close'],
        name='Training data Closing price'
    )
)

fig_preds.add_trace(
    go.Scatter(
        x=valid['Date'],
        y=valid['Close'],
        name='Validation data Closing price'
    )
)

fig_preds.add_trace(
    go.Scatter(
        x=valid['Date'],
        y=valid['Predictions'],
        name='Predicted Closing price'
    )
)

fig_preds.update_layout(legend=dict(
    orientation='h',
    yanchor='bottom',
    y=1,
    xanchor='left',
    x=0)
    , height=600, title_text='Predictions on Validation Data', template='gridon'
)

st.plotly_chart(fig_preds, use_container_width=True)

# metrics
mae = mean_absolute_error(closing_price, valid['Close'])
rmse = np.sqrt(mean_squared_error(closing_price, valid['Close']))
accuracy = r2_score(closing_price, valid['Close']) * 100

# with st.container():
# st.write('#### Metrics')
# col_11, col_22, col_33 = st.columns(3)
# col_11.metric('Absolute error between predicted and actual value', round(mae,2))
# col_22.metric('Root mean squared error between predicted and actual value', round(rmse,2))

# forecasting
real_data = [inputs[len(inputs) - 60:len(inputs + 1), 0]]
real_data = np.array(real_data)
real_data = np.reshape(real_data, (real_data.shape[0], real_data.shape[1], 1))

prediction = model.predict(real_data)
prediction = scaler.inverse_transform(prediction)
st.write('#### Next-Day Forecasting')

with st.container():
    col_111, col_222, col_333 = st.columns(3)
    col_111.metric(f'Closing Price Prediction {symbol} is',
                   f' $ {str(round(float(prediction), 2))}')
    col_222.metric('Accuracy of the model is', f'{str(round(float(accuracy), 2))} %')


