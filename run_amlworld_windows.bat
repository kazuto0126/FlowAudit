@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d %~dp0
if exist data\raw\HI-Small_Trans.csv goto run
echo [FlowAudit] 找不到 data\raw\HI-Small_Trans.csv
echo 請先到 Kaggle 下載資料集 IBM Transactions for Anti Money Laundering 裡的 HI-Small_Trans.csv，放到 data\raw 資料夾：
echo https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml
pause
exit /b 1

:run
echo [FlowAudit] 安裝套件（第一次需要幾分鐘）...
python -m pip install -r requirements.txt
echo [FlowAudit] 開始外部驗證：先跑前 100 萬列，再跑全部資料。
echo [FlowAudit] 可能需要一小時以上，請接上電源、關閉睡眠，不要關掉這個視窗。
python -m flowaudit.amlworld_check
echo.
echo [FlowAudit] 結束。請把 outputs\amlworld_check 整個資料夾傳回。
pause
