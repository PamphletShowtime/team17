#!/usr/bin/env python3
import os
import sys
import numpy as np
from glob import glob
from sklearn.svm import SVC
import joblib

if __name__ == "__main__":
    # Usage: python3 train_model.py <features_dir> <models_dir>
    # features_dirには「features_person01_run01.npy」など、
    # 人IDごとに複数の特徴量ファイルと、それに対応するラベル情報があると仮定します。
    # ここではラベル取得のため、ファイル名からpersonIDを抽出する簡易例を示します。
    # 実際にはCSVやメタ情報ファイルを用意してIDラベルを管理するのがよい。

    if len(sys.argv)<3:
        print("Usage: python train_model.py <features_dir> <models_dir>")
        sys.exit(1)

    features_dir = sys.argv[1]
    models_dir = sys.argv[2]

    if not os.path.exists(models_dir):
        os.makedirs(models_dir)

    # 例: features_ フォーマットのnpファイルを全部読み込む
    feature_files = glob(os.path.join(features_dir, "features_*.npy"))

    X = []
    y = []
    for ff in feature_files:
        feat = np.load(ff)
        # ファイル名からpersonIDを推定する例
        # 実際には person_01, person_02のような名前で管理するのが望ましい。
        # ここでは簡易例: "features_person01_run01.npy" からperson01をIDとして取り出す想定
        fname = os.path.basename(ff)
        # 例えばファイル名に "person01" を含めているとして、
        # 正規表現かsplitでIDを抽出
        # ここでは仮に "person01" という文字列が入っていると想定
        if "person01" in fname:
            label = 1
        elif "person02" in fname:
            label = 2
        else:
            label = 0  # 不明なら0など
        X.append(feat)
        y.append(label)

    X = np.array(X)
    y = np.array(y)

    # SVMで学習
    clf = SVC(probability=True)
    clf.fit(X, y)

    model_path = os.path.join(models_dir, "classifier.pkl")
    joblib.dump(clf, model_path)
    print("Model saved to:", model_path)

