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

df=pd.read_csv("studentdataset.csv")


train_set,test_set=train_test_split(df,test_size=0.2,random_state=42)
student = train_set.drop("final_exam_score", axis=1)
student_labels = train_set["final_exam_score"].copy()
print(student_labels)

student_num=student.select_dtypes(include=["number"])
student_cat=student.select_dtypes(include=["object"])

num_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="median")),
    ("scaler",StandardScaler())
])
cat_pipeline=Pipeline([
    ("imputer",SimpleImputer(strategy="most_frequent")),
    ("encode",OneHotEncoder(handle_unknown="ignore"))
])

full_pipeline=ColumnTransformer([
    ("num",num_pipeline,student_num.columns),
    ("cat",cat_pipeline,student_cat.columns)
])
student_prepare=full_pipeline.fit_transform(student)

# Linear Regression
lin_reg = LinearRegression()
lin_reg.fit(student_prepare, student_labels)

lin_rmse = -cross_val_score(
    lin_reg,
    student_prepare,
    student_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("Linear Regression")
print(pd.Series(lin_rmse).describe())


dec_reg = DecisionTreeRegressor()
dec_reg.fit(student_prepare, student_labels)

dec_rmse = -cross_val_score(
    dec_reg,
    student_prepare,
    student_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("Decision Tree")
print(pd.Series(dec_rmse).describe())



ran_for = RandomForestRegressor()
ran_for.fit(student_prepare, student_labels)

ran_rmse = -cross_val_score(
    ran_for,
    student_prepare,
    student_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("Random Forest")
print(pd.Series(ran_rmse).describe())


xg_reg = XGBRegressor()
xg_reg.fit(student_prepare, student_labels)

xg_rmse = -cross_val_score(
    xg_reg,
    student_prepare,
    student_labels,
    scoring="neg_root_mean_squared_error",
    cv=10
)

print("XGBoost")
print(pd.Series(xg_rmse).describe())