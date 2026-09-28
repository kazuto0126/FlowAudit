#!/usr/bin/env bash
# FlowAudit：IBM AMLworld 外部驗證（一鍵執行）
cd "$(dirname "$0")"
if [ ! -f data/raw/HI-Small_Trans.csv ]; then
  echo "[FlowAudit] 找不到 data/raw/HI-Small_Trans.csv"
  echo "請先到 Kaggle 下載資料集 IBM Transactions for Anti Money Laundering 裡的 HI-Small_Trans.csv，放到 data/raw 資料夾："
  echo "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"
  exit 1
fi
echo "[FlowAudit] 安裝套件（第一次需要幾分鐘）..."
python3 -m pip install -r requirements.txt
echo "[FlowAudit] 開始外部驗證：先跑前 100 萬列，再跑全部資料。"
echo "[FlowAudit] 可能需要一小時以上，請接上電源、關閉睡眠，不要關掉這個視窗。"
python3 -m flowaudit.amlworld_check
echo
echo "[FlowAudit] 結束。請把 outputs/amlworld_check 整個資料夾傳回。"
