"""
Write oversampling functions to form a balanced dataset.
input: "trainingdata-withaddedfeatures-withlexiconfeatures.csv"
output: upsampled data for all minority classes.
"""
from sklearn.utils import resample
import imblearn
import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, SGDClassifier, Perceptron, RidgeClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC
from sklearn.metrics import f1_score
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.model_selection import StratifiedKFold,cross_val_score,cross_val_predict

def resampledata(input_path,outout_path, cats):
    df = pd.read_csv(input_path, header=0)
    #print(list(df.columns.values)) #Prints column names.
    #print(df['post_category'].value_counts()) #710, 295, 137, 40
   # print(df['post_category']=="crisis")
    df_green = df[df['post_category']=="green"]
    df_crisis = df[df['post_category']=="crisis"]
    df_amber =  df[df['post_category']=="amber"]
    df_red =  df[df['post_category']=="red"]
    df_crisis_upsampled = resample(df_crisis,replace=True, n_samples=len(df_green), random_state=123) # reproducible results
    df_red_upsampled = resample(df_red,replace=True, n_samples=len(df_green), random_state=123) # reproducible results
    df_amber_upsampled = resample(df_amber,replace=True, n_samples=len(df_green), random_state=123) # reproducible results
    print(len(df_crisis_upsampled))
    print(len(df_amber_upsampled))
    print(len(df_red_upsampled))
    print(len(df_green))
    df_full_upsampled = pd.concat([df_green,df_red_upsampled,df_amber_upsampled,df_crisis_upsampled])
    df_full_upsampled.to_csv(output_path)
    return df_full_upsampled

def train_models(df_full):
    y = df_full['post_category']
    X = df_full.drop('post_category', axis=1)
    classifiers = [LogisticRegression(C=0.1, max_iter=500), SVC(kernel='linear', probability=True), SGDClassifier(), RidgeClassifier(), Perceptron(), MLPClassifier()]
    for classifier in classifiers:
        print(classifier)
        cross_val = cross_val_score(classifier, X, y, cv=StratifiedKFold(10), n_jobs=1)
        print(cross_val)
        print(sum(cross_val)/float(len(cross_val)))

cats = ["green", "amber", "red", "crisis"]
input_path = "../../data/trainingdata-forresampling-allfeatures.csv"
output_path = "../../data/trainingdata-resampled-py-allfeatures.csv"

df = resampledata(input_path,output_path,cats)
train_models(df)

