#!/usr/bin/env python3
import sys
import os
import numpy as np
from sklearn.svm import SVC
import joblib

if __name__ == "__main__":
    # Usage: python infer.py <model_file> <features_file>
    # model_file: train_model.pyで作成したclassifier.pklなど
    # features_file: extract_features.pyで作成したfeatures_xxx.npy
    if len(sys.argv)<3:
        print("Usage: python infer.py <model_file> <features_file>")
        sys.exit(1)

    model_file = sys.argv[1]
    features_file = sys.argv[2]

    clf = joblib.load(model_file)
    feat = np.load(features_file).reshape(1,-1)  # 1サンプル分の特徴量

    pred = clf.predict(feat)
    print("Predicted Person ID:", pred[0])

