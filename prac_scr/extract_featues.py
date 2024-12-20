#!/usr/bin/env python3
import os
import sys
import numpy as np
import math
from glob import glob

# 使用する関節インデックス（COCOモデルの場合は18点、BODY_25なら25点）
# ここではCOCOモデル(18 keypoints)を想定し、インデックスは0~17
# [鼻0, 目1,目2,耳3,耳4,肩5,肩6,肘7,肘8,手首9,10,腰11,12,膝13,14,足首15,16]
# 実際にはご利用のモデルに応じて修正してください。
KEYPOINT_NUM = 18

def compute_angle(v1, v2):
    # v1,v2は2次元ベクトル
    dot = v1[0]*v2[0] + v1[1]*v2[1]
    mag1 = math.sqrt(v1[0]**2+v1[1]**2)
    mag2 = math.sqrt(v2[0]**2+v2[1]**2)
    if mag1*mag2 == 0:
        return 0.0
    cos_angle = dot / (mag1*mag2)
    # 浮動小数誤差対策
    cos_angle = min(max(cos_angle, -1.0), 1.0)
    angle = math.acos(cos_angle)
    return angle

def extract_features_from_frames(frames_candidates, frames_subsets):
    # frames_candidates, frames_subsetsはフレームごとの(candidates, subsets)データを格納したリスト
    # frames_candidates[i] はあるフレームにおけるcandidate配列
    # frames_subsets[i] はあるフレームにおけるsubset配列

    # 各フレームから主要人物（subsetでスコア最大）を1名抽出
    person_keypoints = []
    for cand, sub in zip(frames_candidates, frames_subsets):
        if len(sub) == 0:
            # 人物検出なしの場合はゼロ埋め
            person_keypoints.append(np.zeros((KEYPOINT_NUM,2)))
            continue
        # スコア最大の人物インデックスを特定
        max_score_idx = np.argmax(sub[:, -2]) # subsetは最後から2番目ぐらいにスコアがある想定
        person = sub[max_score_idx]
        # personは [kp1,kp2,...(各kpへのcandidateインデックス),スコア,人キーポイント数]のような形式
        # candidateインデックスから座標取得
        keypoints_2d = []
        for k in range(KEYPOINT_NUM):
            c_idx = int(person[k])
            if c_idx == -1:
                # 該当キーポイントが見つからない場合は(0,0)で埋める
                keypoints_2d.append([0,0])
            else:
                # candidate: (x,y,confidence)
                x,y,confidence = cand[c_idx][:3]
                keypoints_2d.append([x,y])
        keypoints_2d = np.array(keypoints_2d)
        person_keypoints.append(keypoints_2d)

    person_keypoints = np.array(person_keypoints) # shape: (frame_num, KEYPOINT_NUM, 2)

    # 正規化(d)
    # 例：首(keypoint=1)、腰(keypoint=11)などから身長代わりの尺度を計算し、その距離で割る
    # ここはモデル定義によって異なるので適宜修正してください。
    # 例えば肩(5)と腰(11)の垂直距離を基準とするなど
    base_length = 1.0
    if KEYPOINT_NUM > 11:
        # 仮に肩(5)と腰(11)の距離を基準にする
        # 最初のフレームで計算（なければ別のフレーム）
        ref_frame = person_keypoints[0]
        if np.any(ref_frame[5]) and np.any(ref_frame[11]):
            base_length = np.linalg.norm(ref_frame[5]-ref_frame[11])
    if base_length < 1e-6:
        base_length = 1.0
    person_keypoints /= base_length

    # 各関節間距離(a)と角度(b)の例
    # 距離例：肩(5)-肘(7), 肘(7)-手首(9), 腰(11)-膝(13), 膝(13)-足首(15)
    # 角度例：肘角度(肩-肘-手首)、膝角度(腰-膝-足首)
    # 時系列(c)：フレーム間の速度・加速度
    # 以下は簡易例で本来はもっと多くの特徴を抽出します。

    distances = []
    angles = []
    for frame_kp in person_keypoints:
        # 距離
        dist_shoulder_elbow = np.linalg.norm(frame_kp[5]-frame_kp[7])
        dist_elbow_wrist = np.linalg.norm(frame_kp[7]-frame_kp[9])
        dist_hip_knee = np.linalg.norm(frame_kp[11]-frame_kp[13])
        dist_knee_ankle = np.linalg.norm(frame_kp[13]-frame_kp[15])
        distances.extend([dist_shoulder_elbow, dist_elbow_wrist, dist_hip_knee, dist_knee_ankle])

        # 角度: 肘角度(肩-肘-手首)
        v_shoulder_elbow = frame_kp[7]-frame_kp[5]
        v_elbow_wrist = frame_kp[9]-frame_kp[7]
        elbow_angle = compute_angle(v_shoulder_elbow,v_elbow_wrist)

        # 膝角度(腰-膝-足首)
        v_hip_knee = frame_kp[13]-frame_kp[11]
        v_knee_ankle = frame_kp[15]-frame_kp[13]
        knee_angle = compute_angle(v_hip_knee,v_knee_ankle)

        angles.extend([elbow_angle, knee_angle])

    # 時系列特徴：フレーム間の差分(速度・加速度)
    # フレーム数が3なら、フレーム0->1, 1->2の変化を計算
    velocities = []
    accelerations = []
    dt = 0.2 # 0.2秒ごと
    for i in range(person_keypoints.shape[0]-1):
        v = (person_keypoints[i+1]-person_keypoints[i]) / dt
        velocities.extend(v.flatten())
    for i in range(person_keypoints.shape[0]-2):
        a = ((person_keypoints[i+2]-person_keypoints[i+1]) - (person_keypoints[i+1]-person_keypoints[i])) / (dt*dt)
        accelerations.extend(a.flatten())

    # 全特徴量をまとめる
    feature_vector = distances + angles + velocities + accelerations
    feature_vector = np.array(feature_vector, dtype=np.float32)
    return feature_vector

if __name__ == "__main__":
    # 引数: 入力結果ディレクトリ, 出力先ディレクトリ
    # 例: python3 extract_features.py results/ features/
    if len(sys.argv) < 3:
        print("Usage: python extract_features.py <input_results_dir> <output_features_dir>")
        sys.exit(1)
    input_dir = sys.argv[1]
    output_dir = sys.argv[2]

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # resultsディレクトリ内のcandidate_*, subset_*ファイルをフレーム順に読み込み
    candidate_files = sorted(glob(os.path.join(input_dir, "candidate_*.npy")))
    subset_files = sorted(glob(os.path.join(input_dir, "subset_*.npy")))

    # フレームが3枚程度想定
    frames_candidates = [np.load(cf) for cf in candidate_files]
    frames_subsets    = [np.load(sf) for sf in subset_files]

    # 特徴量抽出
    features = extract_features_from_frames(frames_candidates, frames_subsets)

    # 出力ファイル名（input_dir名を用いるなど工夫可能）
    # シンプルにtimestamp付きにする
    base_name = os.path.basename(input_dir.strip("/"))
    feature_path = os.path.join(output_dir, f"features_{base_name}.npy")
    np.save(feature_path, features)
    print("Features saved to:", feature_path)
