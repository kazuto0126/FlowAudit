@echo off
chcp 65001 >nul
set PYTHONUTF8=1
cd /d %~dp0
echo [FlowAudit] 安裝套件（第一次需要幾分鐘）...
python -m pip install -r requirements.txt
if not exist outputs\tw_sim\results.pkl (
  echo [FlowAudit] 第一次執行：訓練模型與全查（約 2~3 分鐘）...
  python -m flowaudit.pipeline
)
echo [FlowAudit] 啟動儀表板，幾秒後瀏覽器會自動開啟 http://localhost:8501
echo [FlowAudit] 若沒有自動開啟，請手動在瀏覽器輸入 http://localhost:8501 ；要結束請關閉這個視窗
start "" /min cmd /c "timeout /t 8 /nobreak >nul & start "" http://localhost:8501"
python -m streamlit run app.py --server.headless true --browser.gatherUsageStats false
pause