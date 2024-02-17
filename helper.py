import pandas as pd
import pickle
from sklearn import preprocessing


def load_dataset(multiclass=False, normalize=False):
    df = pd.read_csv("train.csv")
    data, labels = df.drop("class", axis=1), df["class"].copy()
    if not multiclass:
        labels = labels.apply(lambda x: 0 if x == 0 else 1)
    if normalize:
        data = preprocessing.minmax_scale(data)
    return data, labels

def save_model(model, path):
    with open(path, "wb") as f:
        pickle.dump(model, f)
