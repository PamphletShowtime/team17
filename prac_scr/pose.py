import os
import cv2
import copy
from src import util
from src.body import Body

# 入力画像ディレクトリと出力ディレクトリのパス
INPUT_DIR = "./pytorch-openpose/images/data"
OUTPUT_DIR = "./pytorch-openpose/result/data"

if __name__ == "__main__":
    # モデルの読み込み
    body_estimation = Body('/home/jetson/pytorch-openpose/model/body_pose_model.pth')

    # 入力ディレクトリ内のすべての画像ファイルを取得
    image_files = [f for f in os.listdir(INPUT_DIR) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]

    # 出力ディレクトリを作成（存在しない場合）
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # 画像ファイルごとに骨格検出を実行
    for image_file in image_files:
        target_image_path = os.path.join(INPUT_DIR, image_file)
        oriImg = cv2.imread(target_image_path)  # B,G,R order

        if oriImg is None:
            print(f"Failed to read image: {target_image_path}")
            continue

        # 骨格検出
        candidate, subset = body_estimation(oriImg)

        # 結果を描画
        canvas = copy.deepcopy(oriImg)
        canvas = util.draw_bodypose(canvas, candidate, subset)

        # 出力ファイルのパスを生成して保存
        basename_name = os.path.splitext(os.path.basename(target_image_path))[0]
        result_image_path = os.path.join(OUTPUT_DIR, f"pose_{basename_name}.jpg")
        cv2.imwrite(result_image_path, canvas)
        print(f"Processed and saved: {result_image_path}")

    print("All images processed.")
