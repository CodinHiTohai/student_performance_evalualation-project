import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import root_mean_squared_error
from xgboost import XGBRegressor
from sklearn.model_selection import cross_val_score
import os
import joblib

MODEL_FILE="model.pkl"
PIPELINE_FILE="pipeline.pkl"
def build_pipeline(num_attribs,cat_attribs):
        
    num_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="median")),
        ("scaler",StandardScaler())
    ])
    cat_pipeline=Pipeline([
        ("imputer",SimpleImputer(strategy="most_frequent")),
        ("encode",OneHotEncoder(handle_unknown="ignore"))
    ])

    full_pipeline=ColumnTransformer([
        ("num",num_pipeline,num_attribs.columns),
        ("cat",cat_pipeline,cat_attribs.columns)
    ])
    return full_pipeline
if not os.path.exists(MODEL_FILE):
        
    df=pd.read_csv("studentdataset.csv")


    train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
    test_features = test_set.drop("final_exam_score", axis=1)
    test_features.to_csv("input.csv", index=False)
    

    student_feature = train_set.drop("final_exam_score", axis=1)
    student_labels = train_set["final_exam_score"].copy()


    student_num = student_feature.select_dtypes(include=["number"])
    student_cat = student_feature.select_dtypes(include=["object"])
    pipeline=build_pipeline(student_num,student_cat)
    student_prepare=pipeline.fit_transform(student_feature)
    print(student_prepare)
    model=LinearRegression()
    model.fit(student_prepare,student_labels)
    joblib.dump(model,MODEL_FILE)
    joblib.dump(pipeline,PIPELINE_FILE)
    print("the model is trained successful")
else:
    model=joblib.load(MODEL_FILE)
    pipeline=joblib.load(PIPELINE_FILE)
    input_data=pd.read_csv('input.csv')
    transformed_input = pipeline.transform(input_data)
    predictions=model.predict(transformed_input)
    input_data['final_exam_score']=predictions
    input_data.to_csv("output.csv",index=False)

print("inferece is complete ,results save to output.csv")
