"""產生 A1 背板海報（594 × 841 mm）PDF 與預覽圖。

數字一律從 outputs/tw_sim/evaluation.json 等結果檔讀取；圖表用報告書產生程式輸出的 SVG（向量，放大列印不失真）、
截圖用 2 倍解析度的 PNG。先執行 docs/報告書/產生程式/make_assets.py，再執行本檔。
需要：pip install playwright（使用本機的 Microsoft Edge 列印成 PDF）。
海報不得出現姓名及學校名稱；也不放任何連結或 QR code。
"""
import json
from pathlib import Path

from PIL import Image
from playwright.sync_api import sync_playwright

HERE = Path(__file__).parent
ROOT = HERE.parents[1]
A = HERE.parent / "報告書" / "產生程式" / "assets"
OUT_PDF = HERE / "FlowAudit_海報_A1.pdf"
OUT_PNG = HERE / "海報預覽.png"

ev = json.loads((ROOT / "outputs/tw_sim/evaluation.json").read_text(encoding="utf-8"))
mt = json.loads((ROOT / "outputs/tw_sim/metrics.json").read_text(encoding="utf-8"))
aml = json.loads((HERE.parent / "報告書" / "產生程式" / "amlworld_all.json").read_text(encoding="utf-8"))
T = ev["temporal"]["summary"]["100"]
F = ev["fair"]
n_pos = F["n_mules"]
eq = {r["k"]: r for r in F["by_k"]}[n_pos]
r80 = next(r for r in F["equal_recall"] if r["recall"] == 0.8)
d_ai = T["ai"]["prevented_by_delay"]
G, TR = ev["graph"]["groups"], ev["graph"]["trace"]["downstream_rule_hit"]
learn = {k: [v["learning"][k] for v in ev["unseen"]["schemes"].values()] for k in ("0", "10")}
fb, fb2 = ev["fallback"]["tw_sim"]["system"], ev["fallback"]["tw_sim_seed2"]["system"]
s2 = mt["external_validation"]["tw_sim_seed2"]
am, am_sum = aml["metrics"], aml["summary"]
am_at = {r["k"]: r for r in am["model"]["at_k"]}
n_ml = am_sum["n_labeled_positive"]
cycle_recall = ev["errors"]["recall_by_scheme"]["循環交易"]


def wan(x):
    return f"{x / 1e4:,.0f}"


def img(name):
    return (A / name).as_uri()


HTML = f"""<!doctype html>
<html lang="zh-Hant"><head><meta charset="utf-8"><title>FlowAudit 金流偵探 海報</title>
<style>
@page {{ size: 594mm 841mm; margin: 0 }}
* {{ box-sizing: border-box }}
html, body {{ margin: 0; padding: 0 }}
body {{ width: 594mm; height: 841mm; font-family: "Microsoft JhengHei", sans-serif; color: #1f1f1f; background: #ffffff;
        -webkit-print-color-adjust: exact; print-color-adjust: exact }}
.poster {{ width: 594mm; height: 841mm; padding: 15mm 18mm 11mm; display: flex; flex-direction: column; gap: 6mm; overflow: hidden }}
header {{ text-align: center; padding-bottom: 4mm; border-bottom: 2mm solid #2a78d6 }}
.event {{ font-size: 22pt; color: #52514e; letter-spacing: 0.12em }}
h1 {{ margin: 2mm 0 0; font-size: 104pt; line-height: 1.08; font-weight: 700 }}
h1 .en {{ color: #2a78d6 }}
.sub {{ font-size: 33pt; font-weight: 700; margin-top: 1mm }}
.tag {{ margin-top: 2mm; font-size: 24pt; color: #52514e }}
.cols {{ flex: 1; display: grid; grid-template-columns: 1fr 1fr; gap: 7mm; min-height: 0 }}
.col {{ display: flex; flex-direction: column; gap: 6mm; min-height: 0 }}
.panel {{ border-radius: 3mm; padding: 4.5mm 5.5mm 5mm; background: #f5f4f1 }}
.panel.ai {{ background: #e9f1fb }}
.panel.rule {{ background: #fdf1ea }}
h2 {{ margin: 0 0 3mm; font-size: 30pt; font-weight: 700; display: flex; align-items: center; gap: 3mm }}
h2 .no {{ display: inline-block; width: 12mm; height: 12mm; border-radius: 50%; background: #2a78d6; color: #fff;
          font-size: 19pt; text-align: center; line-height: 12mm; flex: none }}
p {{ font-size: 20pt; line-height: 1.45; margin: 0 }}
p + p {{ margin-top: 2mm }}
b.ai, .ai b, span.ai {{ color: #2a78d6 }}
b.rule, span.rule {{ color: #c4501d }}
.stats {{ display: grid; grid-template-columns: repeat(3, 1fr); gap: 4mm; margin-bottom: 3mm }}
.stat {{ background: #fff; border-radius: 2.5mm; padding: 3mm 4mm }}
.stat .n {{ font-size: 40pt; font-weight: 700; color: #2a78d6; line-height: 1.1 }}
.stat .n.rule {{ color: #c4501d }}
.stat .l {{ font-size: 16pt; color: #52514e; line-height: 1.35 }}
.fig {{ display: block; width: 100%; background: #fff; border-radius: 2mm }}
.shot {{ display: block; width: 100%; border: 0.4mm solid #c9c8c4; border-radius: 2mm; background: #fff }}
.cap {{ font-size: 15pt; color: #52514e; margin-top: 1.5mm; line-height: 1.35 }}
.two {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4mm }}
.kpis {{ display: grid; grid-template-columns: 1fr 1fr; gap: 4mm; margin-bottom: 3mm }}
.kpi {{ background: #fff; border-radius: 2.5mm; padding: 3mm 4mm }}
.kpi .v {{ font-size: 36pt; font-weight: 700; line-height: 1.1 }}
.kpi .l {{ font-size: 16pt; color: #52514e; line-height: 1.35 }}
.fix {{ display: grid; grid-template-columns: 34mm 1fr; gap: 2mm 4mm; align-items: start }}
.fix .k {{ font-size: 18.5pt; font-weight: 700; color: #c4501d; line-height: 1.4 }}
.fix .v {{ font-size: 18.5pt; line-height: 1.4 }}
.chips {{ display: flex; flex-wrap: wrap; gap: 2.5mm }}
.chip {{ font-size: 17pt; background: #fff; border: 0.4mm solid #2a78d6; border-radius: 10mm; padding: 1mm 4mm }}
footer {{ font-size: 13.5pt; color: #52514e; text-align: center; line-height: 1.4 }}
</style></head>
<body><div class="poster">

<header>
  <div class="event">2026 法遵科技與電腦稽核專題競賽</div>
  <h1><span class="en">FlowAudit</span> 金流偵探</h1>
  <div class="sub">以資金流向圖與可解釋 AI 偵測人頭帳戶的持續稽核系統</div>
  <div class="tag">每月全查交易 ・ 規則與 AI 分工 ・ 說得出為什麼可疑 ・ 自動產出稽核工作底稿</div>
</header>

<div class="cols">
<div class="col">

  <section class="panel">
    <h2><span class="no">1</span>為什麼需要</h2>
    <div class="stats">
      <div class="stat"><div class="n">893 億元</div><div class="l">2025 年全台詐騙財損</div></div>
      <div class="stat"><div class="n">14.85 萬</div><div class="l">全台警示帳戶（2026 年 1 月）</div></div>
      <div class="stat"><div class="n rule">{F['any_rule']['normal']:,}</div>
        <div class="l">命中任一規則的 {F['any_rule']['n']:,} 個帳戶中的正常帳戶（本研究仿真資料）</div></div>
    </div>
    <p>詐騙款項經第一層、第二層人頭帳戶在數小時內轉出，帳戶卻多在警方通報後才被警示；依固定門檻的規則又會把團購主、
       發薪公司等正常帳戶一起列為可疑。</p>
  </section>

  <section class="panel">
    <h2><span class="no">2</span>系統怎麼運作</h2>
    <img class="fig" src="{img('fig_architecture.svg')}">
    <p style="margin-top:3mm"><b class="ai">全查</b>：8 條紅旗規則逐條對應調查局態樣與金管會辦法，附證據文字。
       <b class="ai">AI 排序</b>：57 項特徵、XGBoost，只用已被警示的帳戶訓練。
       <b class="ai">三份名單</b>：AI 名單 90%、規則名單 10%，另加資金循環保底名單。</p>
  </section>

  <section class="panel" style="flex:1">
    <h2><span class="no">3</span>系統展示</h2>
    <img class="shot" src="{img('shot_investigate.png')}">
    <div class="cap">帳戶調查：左為 AI 判斷依據（紅色推高風險、藍色降低風險），右為資金往來網路</div>
    <img class="shot" style="margin-top:3mm" src="{img('shot_evidence.png')}">
    <div class="cap">每條命中的規則都附態樣代碼與交易證據，可一鍵產生可疑交易分析報告與 Excel 工作底稿</div>
    <p style="margin-top:4mm"><b class="ai">使用情境</b>：法遵與洗錢防制單位每月依覆核名單調查、決定是否列警示或申報；
       內部稽核以系統獨立全查，檢驗既有監控機制是否漏掉可疑帳戶；會計師事務所也能對客戶提供的交易資料全查。</p>
  </section>

</div>
<div class="col">

  <section class="panel ai">
    <h2><span class="no">4</span>成效（台灣情境仿真資料）</h2>
    <div class="kpis">
      <div class="kpi"><div class="v"><span class="ai">{T['ai']['unique_mules']}</span> 對 <span class="rule">{T['rules']['unique_mules']}</span></div>
        <div class="l">每月只覆核 100 個帳戶，6 個月找到的人頭帳戶（AI 對規則）</div></div>
      <div class="kpi"><div class="v"><span class="ai">{T['ai']['caught_before_alert']} 個</span>、早 {T['ai']['lead_days_mean']:.0f} 天</div>
        <div class="l">比警方通報更早發現的人頭帳戶與平均提早天數</div></div>
      <div class="kpi"><div class="v"><span class="ai">{eq['ai']['mules']:.0f}</span> 對 <span class="rule">{eq['rules']['mules']:.0f}</span></div>
        <div class="l">覆核同樣 {n_pos} 個帳戶找到的人頭帳戶；誤查的正常帳戶 {eq['ai']['normal']:.0f} 對 {eq['rules']['normal']:.0f}</div></div>
      <div class="kpi"><div class="v"><span class="ai">{r80['ai']['reviewed']:,.0f}</span> 對 <span class="rule">{r80['rules']['reviewed']:,.0f}</span></div>
        <div class="l">要找回八成人頭帳戶，各需覆核的帳戶數</div></div>
    </div>
    <img class="fig" src="{img('fig_equal_volume.svg')}">
    <p style="margin-top:2.5mm">潛在攔阻金額（確認當天凍結，仿真情境上限）{wan(d_ai['0'])} 萬元；調查延遲 3 天降為 {wan(d_ai['3'])} 萬元：
       提早發現之外，也要及早處置。</p>
  </section>

  <section class="panel rule">
    <h2><span class="no">5</span>AI 何時會失效，系統怎麼補</h2>
    <div class="fix">
      <div class="k">導入初期</div><div class="v">警示資料太少 → <b>冷啟動</b>：先用參考資料訓練的模型，6 個月找到 {T['cold']['unique_mules']} 個（只用自家資料 {T['ai']['unique_mules']} 個）</div>
      <div class="k">新詐騙手法</div><div class="v">沒見過的手法只找回 {min(learn['0']):.0%}～{max(learn['0']):.0%} → <b>雙名單＋每月重新訓練</b>：累積 10 個警示帳戶後回升到 {min(learn['10']):.0%}～{max(learn['10']):.0%}</div>
      <div class="k">循環交易</div><div class="v">很少被通報，AI 只找到 {cycle_recall:.0%} → <b>規則保底名單</b>：{fb['rule_list_cycle'] + fb['fallback_cycle']} 個全部找到（另一份資料 {fb2['rule_list_cycle'] + fb2['fallback_cycle']} 個也全找到）</div>
    </div>
  </section>

  <section class="panel">
    <h2><span class="no">6</span>資金流向圖：用在調查</h2>
    <div style="display:flex; gap:5mm; align-items:flex-start">
    <p style="flex:1">系統切出的 <b class="ai">{G['n_groups']} 個疑似集團，{G['single_fraud_group']} 個都只對應單一真實詐騙集團</b>。
       從已警示帳戶沿資金往下游追一層、再篩命中紅旗規則的帳戶：<b class="ai">{TR['n']} 個帳戶中有 {TR['mules']} 個是警方尚未通報的人頭帳戶</b>，
       其中 {TR['cycle_mules']} 個是 AI 不易找到的循環交易。</p>
    <div style="width:112mm; flex:none"><img class="shot" src="{img('shot_group.png')}">
      <div class="cap">疑似集團分析：集團資金網路與成員</div></div>
    </div>
  </section>

  <section class="panel">
    <h2><span class="no">7</span>誠實驗證</h2>
    <img class="fig" src="{img('fig_external.svg')}">
    <p style="margin-top:2.5mm"><b>仿真穩健性測試</b>：另一份仿真資料不重新訓練，AI {s2['model_pr_auc']:.3f}、規則 {s2['rules_pr_auc']:.3f}。
       <b>IBM AMLworld</b>（公開的合成資料，以其標籤重新訓練）：AI {am['model']['pr_auc']:.3f}，約為隨機的 {am['model']['pr_auc'] / am['positive_rate']:.0f} 倍；
       覆核 {n_ml:,} 名時找回 {am_at[n_ml]['recall']:.0%}。<b>真實銀行資料</b>：尚待驗證。</p>
  </section>

  <section class="panel" style="flex:1">
    <h2><span class="no">8</span>應用對象與導入</h2>
    <div class="chips">
      <span class="chip">銀行法遵與洗錢防制</span><span class="chip">內部稽核</span><span class="chip">電子支付、純網銀</span>
      <span class="chip">信用合作社、農漁會信用部</span><span class="chip">會計師事務所</span><span class="chip">主管機關</span>
    </div>
    <p style="margin-top:3mm">完全離線運作；導入三階段：以歷史資料重新訓練驗證 → 與現行系統平行運作 → 納入每月作業並建立模型治理。</p>
  </section>

</div>
</div>

<footer>資料說明：主要結果來自依公開法規、統計與報導設計的台灣情境合成資料，不含真實個人資料；AMLworld 為 IBM 公開的合成資料。
方法、數字與限制詳見企劃書。</footer>
</div></body></html>
"""


def main():
    html_path = HERE / "poster.html"
    html_path.write_text(HTML, encoding="utf-8")
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        pg = b.new_page(viewport={"width": 2245, "height": 3179})  # A1 在 96 dpi 下的 CSS 像素
        pg.goto(html_path.as_uri())
        pg.wait_for_load_state("networkidle")
        overflow = pg.evaluate("""() => [...document.querySelectorAll('.col')].map(c => c.scrollHeight - c.clientHeight)""")
        pg.pdf(path=str(OUT_PDF), width="594mm", height="841mm", print_background=True, prefer_css_page_size=True)
        pg.screenshot(path=str(OUT_PNG), full_page=False)
        b.close()
    im = Image.open(OUT_PNG)
    im.resize((im.width // 2, im.height // 2), Image.LANCZOS).save(OUT_PNG, optimize=True)
    html_path.unlink()
    print("saved", OUT_PDF.name, OUT_PNG.name, "| 欄位溢出（px，需為 0 以下）：", overflow)


if __name__ == "__main__":
    main()
