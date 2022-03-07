from pandas._config.config import options
import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import datetime
import pickle
import random
import seaborn as sns

from utils import temporal_features

sns.set_theme(style="whitegrid", palette="Set2")

# Set configs 
st.set_page_config(
    page_title="ASHRAE Great Energy Predictor III Dashboard",
    initial_sidebar_state="auto",
    menu_items={
        "Get help": None,
        "Report a bug": None,
        "About": None
    }
)

# streamlit runs the whole script every time a change in data occurs (i.e. input change),
# to ensure that the data and model are not loaded on each change, we put them into cache 
@st.cache
def load_data(path):
    with open(path, "rb") as f:
        data = pickle.load(f)
    # data = pd.read_csv(path)
    return data

@st.cache
def load_model(path):
    model = pickle.load(open(path,'rb'))
    return model

# Load data and model 
DATA_PATH = 'C:/dev/studienarbeit/ashrae-energy-prediction-data/pkl/train.pkl'
MODEL_PATH = 'lgbm_regression_pipeline.pkl'

data = load_data(DATA_PATH)
model = load_model(MODEL_PATH)

# Predict function uses dictionary of keyword arguments
@st.cache
def predict(value_dict):
    # df = pd.DataFrame.from_dict(value_dict)
    meter_predictions = {}
    for meter in [0,1,2,3]:
        value_dict["meter"] = [meter]
        meter_predictions[meter] = model.predict(value_dict)
    return meter_predictions

@st.cache(allow_output_mutation=True)
def feature_engineering(values):
    new_value = pd.DataFrame.from_dict(values)
    new_value["building_age"] = 2017.0 - new_value["year_built"]
    temporal_features(new_value)
    new_value = new_value.drop("timestamp", 1)
    return new_value

# dict with current values of all input variables
values = {}

# --- Layout ---

# Dashboard Layout
st.title("Building Energy Predictor")
st.markdown("**Author: Raphael Reimann**")
st.write("This is a Building energy usage prediction dashboard. You can change the input \
         variables to see how it affects the models prediction of a buildings energy usage.")

col0, col1, col2 = st.columns(3)

with col0:
    st.subheader("Building data")
    values["building_id"] = [st.number_input(label="Building ID", value=0)]
    values["site_id"] = [st.number_input(label="Site ID", value=0)]
    values["primary_use"] = [st.selectbox(label="Primary Use", options=data["primary_use"].unique())]
    values["square_feet"] = [st.number_input(label="Square Feet", value=0)]
    values["year_built"] = [st.slider(label="Year Built", min_value=1900, max_value=2017, value=1990)]
    values["floor_count"] = [st.number_input(label="Floor Count", value=0)]
with col1:
    st.subheader("Weather data")
    values["air_temperature"] = [st.slider(label="Air Temperature", min_value=-30.0, max_value=50.0, value=10.0)]
    values["cloud_coverage"] = [st.slider(label="Cloud Coverage", min_value=0, max_value=10, value=0)]
    values["dew_temperature"] = [st.slider(label="Dew Temperature", min_value=-35.0, max_value=30.0, value=10.0)]
    values["precip_depth_1_hr"] = [st.number_input(label="Precip_depth_1_hr", value=0)]
    values["wind_direction"] = [st.slider(label="Wind Direction", min_value=0, max_value=360, value=0)]
with col2:
    st.subheader("Temporal data")
    date = st.date_input(label="Date", value=datetime.date(2017, 6, 26), min_value=datetime.date(2016, 1, 1), max_value=datetime.date(2020, 12, 31))
    time = st.time_input(label="Time", value=datetime.time(8, 45))
    values["timestamp"] = [pd.to_datetime(datetime.datetime.combine(date, time))]   
    
st.markdown("""---""")

# --- Prediction
st.subheader("Model Prediction")
st.write("Here you can see what the model predicts as the meter reading values for each meter based on the inputs above.")
meter_predictions = predict(feature_engineering(values))
metric_cols = st.columns(4)
for i in [0,1,2,3]:
    with metric_cols[i]:
        st.metric(label=f"Meter {i} (kWh)", value=round(float(meter_predictions[i]), 2))

st.markdown("""---""")

# --- Prototype Retrofit analysis
st.title("Prototype Retrofit analysis")
st.markdown("""This is a prototype of how a retrofit analysis of a buildings energy usage could 
         look like in practice. The graph below shows historical energy usage of building and a 
         hypothetical implementation of an energy conservation method (ECM). The month of the implementation 
         of the ECM can be adjusted with the slider below. The machine learning model can then 
         predict the buildings energy usage from the external conditions based on what it has 
         learned from the historical data and the real energy usage after ECM implementation can 
         then be compared with the models prediction. The data in the graph below is not actual 
         predicted data but rather real data taken from our training set for prototyping reasons. 
         More information on retrofit analysis and how the model is built is available in the paper 
         [(Github Link)](https://github.com/raphaelreimann).""")

buildings = data["building_id"].unique()
prototype_input_cols = st.columns(2)
with prototype_input_cols[0]:
    selected_building = st.selectbox(label="Building", options=buildings, index=147)
with prototype_input_cols[1]:
    selected_month = st.slider(label="Month of ECM implementation", min_value=1, max_value=12, value=8)


def make_plot():
    """Make a plot to show retrofit analysis"""
    historical_data = data[(data["building_id"] == selected_building) & (data["timestamp"].dt.month < selected_month)][["timestamp", "meter_reading"]].set_index('timestamp').resample('D').mean()
    predicted_data = data[(data["building_id"] == selected_building) & (data["timestamp"].dt.month >= selected_month)][["timestamp", "meter_reading"]].set_index('timestamp').resample('D').mean()
    new_data = data[(data["building_id"] == selected_building) & (data["timestamp"].dt.month >= selected_month)][["timestamp", "meter_reading"]].set_index('timestamp').resample('D').mean().apply(lambda x: x*random.uniform(0.45, 0.5), axis=1)

    fig, ax = plt.subplots(1, 1, figsize=(10, 4))
    ax.plot(historical_data, label="Historical energy usage", alpha=0.8)
    ax.plot(predicted_data, label="Predicted energy usage", alpha=0.8, linestyle="--")
    ax.plot(new_data, label="Energy usage after conservation measure", alpha=0.8)
    # ax.vlines(x=17000, ymin=0, ymax=100, linestyles='dashed', colors=["k"])
    ax.set_xlabel("Timestamp")
    ax.set_ylabel("Mean meter reading")
    ax.set_title(f"Mean meter reading for building {selected_building}");
    ax.legend()

    st.pyplot(fig)
    
make_plot()