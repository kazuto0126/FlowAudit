# FlowAudit 金流偵探 — 給 Claude Code 的專案說明

以資金流向圖與可解釋 AI 偵測人頭帳戶的持續稽核系統。參加大專院校電腦稽核／RegTech 競賽（2026 年 11 月 30 日投稿截止，12 月 11 日決賽）。

## 常用指令

```bash
pip install -r requirements.txt
python -m flowaudit.simulator                         # 重新產生台灣情境仿真資料（已隨附，通常不用）
python -m flowaudit.pipeline                          # 規則全查 + 特徵 + 模型訓練 + 外部驗證（約 1～2 分鐘）
python -m flowaudit.pipeline --dataset amlsim_fanin_cycle
python -m flowaudit.evaluation                        # 多模型比較、消融、逐月模擬、誤判分析、新手法測試等（約 15 分鐘）
python -m flowaudit.evaluation --stress               # 另外重跑壓力測試（再加約 15 分鐘；未加時沿用上次結果）
python -m flowaudit.amlworld_check                    # IBM AMLworld 外部驗證（需先把 HI-Small_Trans.csv 放到 data/raw/）
python -m streamlit run app.py                        # 儀表板 http://localhost:8501
python -m pytest -q                                   # 測試，改完程式一定要跑
```

在 Windows 上一律用 `python -m ...` 的寫法。

## 專案結構

- `flowaudit/simulator.py`：台灣情境交易仿真器（正常角色、易誤判的正常帳戶、4 種詐騙集團、警示與凍結）
- `flowaudit/data_loader.py`：統一資料格式；`CASH` 是現金存提的對手方，不是帳戶
- `flowaudit/rules.py`：8 條紅旗規則 R1～R8，每條對應調查局疑似洗錢態樣代碼或金管會管理辦法（`RULE_REFS`）
- `flowaudit/features.py`：57 項特徵，分 6 類（`FEATURE_GROUPS`），中文名稱在 `FEATURE_NAMES_ZH`
- `flowaudit/model.py`：XGBoost、集團層級交叉驗證、TreeSHAP、成效指標
- `flowaudit/pipeline.py`：端對端流程、疑似集團偵測、外部驗證、結果存檔（`outputs/<資料集>/`）
- `flowaudit/evaluation.py`：嚴謹評估，輸出 `outputs/tw_sim/evaluation.json`
- `flowaudit/amlworld_check.py`：AMLworld 外部驗證一鍵執行，結果在 `outputs/amlworld_check/`
- `flowaudit/report.py`：可疑交易分析報告（模板／LLM）、Word 匯出、Excel 工作底稿
- `app.py`：Streamlit 儀表板（7 頁）
- `config.yaml`：規則門檻與模型參數；調整門檻改這裡，不要寫死在程式裡
- `docs/仿真設計與參數依據.md`：仿真參數與出處

## 必須遵守的規則

1. **不可有標籤外洩**：特徵與規則只能用交易資料和 `ACCOUNT_ATTRS`（customer_type、open_date、dormant_days_before）。`GROUND_TRUTH_COLS`（role、fraud_group、scheme、mule_layer、mule_source、alert_date）和 `label` 只能用於訓練標籤與評估。
2. **訓練標籤貼近實務**：`model.train_label: alerted` 表示只用「已被警示」的帳戶訓練，評估時用全部真實人頭帳戶。不要改回用全部標籤訓練。
3. **競賽匿名規定**：程式、介面、報告、README 都不能出現姓名、學號、學校名稱。
4. **介面與文件用繁體中文**，用語要讓稽核／會計背景的評審看得懂。
5. **誠實呈現結果**：資料是合成資料，要保留相關說明；不要為了數字好看而調參後只報好的結果。
6. **不要 commit 執行產生的檔案**：`outputs/` 只保留 `outputs/tw_sim/evaluation.json`，其他已被 `.gitignore` 排除；不要把 `.env` 或任何 API 金鑰寫進程式。
7. **改了規則、特徵、模型或仿真器之後**：依序重跑 `pipeline` → `evaluation` → `pytest`，並同步更新 README 的「重點成果」數字和 `docs/仿真設計與參數依據.md`。
8. 圖表顏色沿用 `app.py` 開頭的 `C_AI`、`C_RULE`、`C_RAND`、`C_RISK`，不要另外加顏色。
