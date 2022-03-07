import pandas as pd
import numpy as np
import pickle
import datetime

def load_data(pickle_path, data_path, dataset="train", from_pickle=True, merge_weather_building=True, reduce_mem=True):
    """
    Load datasets and merge with building metadata and weather data 
    """
    assert dataset in ["train", "test", "building", "weather_train", "weather_test"]
    
    # Get dataframe from local pickle file
    if from_pickle:
        with open(f"{pickle_path}/{dataset}.pkl", "rb") as f:
            return pickle.load(f)
    
    # this is the meta data (either train or test set)
    if dataset == "train":
        df = pd.read_csv(f"{data_path}/{dataset}.csv", parse_dates=["timestamp"])        
    if dataset == "test":
        df = pd.read_csv(f"{data_path}/{dataset}.csv", 
                         usecols=["building_id", "meter", "timestamp"], # test set has an additional column row_id which we don"t need
                         parse_dates=["timestamp"])
    if merge_weather_building:
            building = pd.read_csv(f"{data_path}/building_metadata.csv")
            # weather = pd.read_csv(f"{data_path}/weather_{dataset}.csv", parse_dates=["timestamp"])
            weather = pd.read_csv(f"{data_path}/weather_{dataset}.csv")
            # weather = fill_weather_dataset(weather)
            weather["timestamp"] = pd.to_datetime(weather["timestamp"])
            df = df.merge(building, on="building_id", how="left")
            df = df.merge(weather, on=["site_id", "timestamp"], how="left")     
            
    if dataset == "building":
        df = pd.read_csv(f"{data_path}/building_metadata.csv")
    if dataset == "weather_train":
        df = pd.read_csv(f"{data_path}/weather_train.csv", parse_dates=["timestamp"])
    if dataset == "weather_test":
        df = pd.read_csv(f"{data_path}/weather_test.csv", parse_dates=["timestamp"])
    
    if reduce_mem:
        df = reduce_mem_usage(df, verbose=False)
    
    # When changing something in the data loading process, save dataframes as local pkl objects
    if not from_pickle:
        with open(f"{pickle_path}/{dataset}.pkl", "wb") as f:
            pickle.dump(df, f)
    return df

# Function to reduce the DF size
def reduce_mem_usage(df, verbose=True):
    """Taken from https://www.kaggle.com/cereniyim/save-the-energy-for-the-future-1-detailed-eda"""
    numerics = ["int16", "int32", "int64", "float16", "float32", "float64"]
    start_mem = df.memory_usage().sum() / 1024**2    
    for col in df.columns:
        col_type = df[col].dtypes
        if col_type in numerics:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)    
    end_mem = df.memory_usage().sum() / 1024**2
    if verbose: print("Mem. usage decreased to {:5.2f} Mb ({:.1f}% reduction)".format(end_mem, 100 * (start_mem - end_mem) / start_mem))
    return df


def temporal_features(df):
    """Feature Engineering of temporal features (used by ashrae_dashboard.py)"""
    df["day_of_week"] = df["timestamp"].dt.dayofweek.astype(np.int8)
    df["day_of_year"] = df["timestamp"].dt.dayofyear.astype(np.int16)
    df["week_of_year"] = df["timestamp"].dt.isocalendar().week.astype(np.int8)

    df["month"] = df["timestamp"].dt.month.astype(np.int8)
    df["hour"] = df["timestamp"].dt.hour.astype(np.int8)
    df["day"] = df["timestamp"].dt.day.astype(np.int8)
    
