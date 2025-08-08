import pandas as pd
import numpy as np
from sklearn.model_selection import train_split 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.feature_extraction.text import TfidfTransformer
import joblib

df=pd.read_csv("youtube_comments_dataset.csv")

x=df["COMMENT"]
y=df["SENTIMENT"]

vec=Tfidfvectorizer(stop_words='english',max_features=1000)
xx=vec.fit_Transform(x)

x_train,x_test,y_train,y_test=train_test_split(xx,y,test_size=0.2)

ml=RandomForestClassifier()
ml.fit(x_train,y_train)

joblib.dump(ml,"cmts.pkl")
joblib.dump(vec,"vector.pkl")