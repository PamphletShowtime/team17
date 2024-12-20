#!/usr/bin/env python3
import sys
import os
import cv2
import copy
import numpy as np
from src import util
from src.body import Body

if __name__ == "__main__":
    if len(sys.argv) < 3:
        print("Usage: python test.py <input_image> <result_dir>")
        sys.exit(1)

    INPUT_FILE_NAME = sys.argv[1]
    RESULT_DIR = sys.argv[2]

    # ディレクトリがなければ作成
    if not os.path.exists(RESULT_DIR):
        os.makedirs(RESULT_DIR)

    # OpenPoseモデル読み込み
    body_estimation = Body('model/body_pose_model.pth')

    # 画像読み込み
    oriImg = cv2.imread(INPUT_FILE_NAME)
    if oriImg is None:
        print("Failed to read image:", INPUT_FILE_NAME)
        sys.exit(1)

    # 骨格推定
    candidate, subset = body_estimation(oriImg)
    canvas = copy.deepcopy(oriImg)
    canvas = util.draw_bodypose(canvas, candidate, subset)

    basename_name = os.path.splitext(os.path.basename(INPUT_FILE_NAME))[0]
    result_image_path = os.path.join(RESULT_DIR, "pose_" + basename_name + ".jpg")
    cv2.imwrite(result_image_path, canvas)
    print(f"Result saved: {result_image_path}")

    # candidate, subsetをnumpyファイルとして保存
    candidate_path = os.path.join(RESULT_DIR, f"candidate_{basename_name}.npy")
    subset_path = os.path.join(RESULT_DIR, f"subset_{basename_name}.npy")
    np.save(candidate_path, candidate)
    np.save(subset_path, subset)
    print(f"Candidate data saved: {candidate_path}")
    print(f"Subset data saved: {subset_path}")

    # 正常終了
    sys.exit(0)
