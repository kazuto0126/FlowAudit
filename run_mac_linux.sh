#!/usr/bin/env bash
set -e
cd "$(dirname "$0")"
echo "[FlowAudit] 安裝套件（第一次需要幾分鐘）..."
python3 -m pip install -r requirements.txt
if [ ! -f outputs/tw_sim/results.pkl ]; then
  echo "[FlowAudit] 第一次執行：訓練模型與全查（約 2~3 分鐘）..."
  python3 -m flowaudit.pipeline
fi
echo "[FlowAudit] 啟動儀表板：http://localhost:8501"
python3 -m streamlit run app.py
