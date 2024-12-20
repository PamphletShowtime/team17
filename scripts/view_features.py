#!/usr/bin/env python3
import sys
import os
import numpy as np

# view_features.pyはscripts配下にいるため、相対パスでfeaturesディレクトリを指定可能
features_dir = os.path.join(os.path.dirname(__file__), 'features')

if len(sys.argv) < 2:
    # 引数がない場合、scripts/features内の最初のnpyファイルを取得
    npy_files = [f for f in os.listdir(features_dir) if f.endswith('.npy')]
    if not npy_files:
        print("No npy files found in scripts/features directory.")
        sys.exit(1)
    filename = os.path.join(features_dir, npy_files[0])
else:
    filename = sys.argv[1]
    # 引数で渡されたパスが相対パスの場合、scriptsディレクトリからの相対となるため、
    # 必要に応じて絶対パス化するかfeatures_dirとの組み合わせを調整してください。
    # ここでは引数が完全なパスを想定。

data = np.load(filename)
print(data)

