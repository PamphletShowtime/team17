#!/usr/bin/env bash
set -e

# スクリプトが存在するディレクトリ（scripts）からプロジェクトルートへ移動
cd "$(dirname "$0")/.."

# results/result**** ディレクトリをtar.gz形式でアーカイブ
# ここでは、results/result* でマッチする全てのディレクトリをアーカイブします
tar czvf results.tar.gz results/result*

echo "results/result**** ディレクトリをresults.tar.gzに変換しました。"
echo "ファイルはpytorch-openposeディレクトリ直下にあります。"

