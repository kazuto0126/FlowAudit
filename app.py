"""FlowAudit 金流偵探 — Streamlit 稽核儀表板

啟動：streamlit run app.py
"""
from __future__ import annotations

import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import streamlit.components.v1 as components

from flowaudit import data_loader as dl
from flowaudit.data_loader import CASH
from flowaudit.features import FEATURE_GROUPS, FEATURE_NAMES_ZH
from flowaudit.model import cycle_fallback, load_model, review_lists, rule_rank_score
from flowaudit.pipeline import OUT_DIR, analyze, load_config, load_run, save_run
from flowaudit.report import _llm_provider, build_workpaper, collect_facts, generate_report, markdown_to_docx
from flowaudit.rules import RULE_DESCRIPTIONS, RULE_REFS, RULES, rule_availability

st.set_page_config(page_title="FlowAudit 金流偵探", page_icon="🔎", layout="wide")

# 色彩（經色盲安全驗證的參考色盤）：系列色依固定順序；風險等級用狀態色並搭配文字
C_AI, C_RULE, C_RAND = "#2a78d6", "#eb6834", "#1baf7a"
C_RISK = {"高": "#d03b3b", "中": "#ec835a", "低": "#9a9892"}
PLOT_FONT = dict(family="Noto Sans TC, Microsoft JhengHei, PingFang TC, sans-serif", size=13)
CFG = load_config()
DEFAULT_KEY = CFG["data"].get("default_dataset", "tw_sim")


# ---------------------------------------------------------------------------
# 資料載入
# ---------------------------------------------------------------------------
@st.cache_resource(show_spinner="第一次開啟：全查並訓練模型中（約 1～2 分鐘）…")
def get_run(key: str):
    if not (OUT_DIR / key / "results.pkl").exists():
        ds = dl.load_dataset(key, CFG["data"]["base_date"])
        save_run(analyze(ds, CFG, log=lambda *_: None), key)
    return load_run(key)


@st.cache_resource
def get_model(key: str):
    return load_model(get_run(key)["model_path"])


@st.cache_data
def get_eval(key: str, mtime: float):
    p = OUT_DIR / key / "evaluation.json"
    return json.loads(p.read_text(encoding="utf-8")) if p.exists() else None


def eval_for(key):
    p = OUT_DIR / key / "evaluation.json"
    return get_eval(key, p.stat().st_mtime if p.exists() else 0)


def current_key() -> str:
    return st.session_state.get("dataset", DEFAULT_KEY)


def current_run() -> dict:
    return st.session_state.get("uploaded_run") or get_run(current_key())


def is_sim(res: pd.DataFrame) -> bool:
    return "role" in res.columns


def fmt(x: float) -> str:
    return f"{x:,.0f}"


def money(x: float) -> str:
    if abs(x) >= 1e8:
        return f"{x / 1e8:,.2f} 億元"
    if abs(x) >= 1e4:
        return f"{x / 1e4:,.0f} 萬元"
    return f"{x:,.0f} 元"


def kpi(col, label: str, value: str, note: str | None = None):
    col.metric(label, value)
    if note:
        col.caption(note)


def style_fig(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height, font=PLOT_FONT, margin=dict(l=10, r=10, t=40, b=10),
        plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        hoverlabel=dict(font=PLOT_FONT),
    )
    fig.update_xaxes(showgrid=False, linecolor="rgba(128,128,128,0.4)")
    fig.update_yaxes(gridcolor="rgba(128,128,128,0.15)", zeroline=False)
    return fig


# ---------------------------------------------------------------------------
# 網路圖（pyvis，可拖曳縮放）
# ---------------------------------------------------------------------------
def network_html(nodes: dict, edges: list, center: str | None = None, height: int = 520) -> str:
    from pyvis.network import Network

    net = Network(height=f"{height}px", width="100%", directed=True, cdn_resources="in_line", bgcolor="#ffffff")
    net.barnes_hut(gravity=-6000, spring_length=140)
    for n, info in nodes.items():
        lv = info.get("level", "低")
        net.add_node(
            n, label=n, color={"background": C_RISK[lv], "border": "#0b0b0b" if n == center else C_RISK[lv]},
            borderWidth=4 if n == center else 1, size=28 if n == center else 12 + 10 * info.get("score", 0),
            title=f"{n}\nAI 風險分數：{info.get('score', 0):.3f}（{lv}）\n命中規則：{info.get('rules') or '無'}",
            font={"size": 14 if n == center else 11},
        )
    amax = max([e[2] for e in edges] or [1])
    for u, v, amt, cnt in edges:
        net.add_edge(u, v, value=amt, width=1 + 5 * amt / amax, title=f"{u} → {v}\n{cnt} 筆，合計 {fmt(amt)} 元",
                     color="rgba(90,90,90,0.55)", arrows="to")
    html = net.generate_html()
    # 移除 pyvis 範本中的外部 CDN（bootstrap），確保比賽現場離線也能正常顯示
    html = re.sub(r"<link[^>]*bootstrap[^>]*>", "", html)
    html = re.sub(r"<script[^>]*bootstrap[^>]*></script>", "", html)
    return html


def node_info(res, ids):
    return {n: {"score": float(res.at[n, "risk_score"]), "level": res.at[n, "risk_level"], "rules": res.at[n, "rule_list"]}
            for n in ids}


def ego_graph(run: dict, acct: str, max_nbrs: int = 40):
    tx, res = run["tx"], run["results"]
    rel = tx[((tx["src"] == acct) | (tx["dst"] == acct)) & (tx["src"] != CASH) & (tx["dst"] != CASH)]
    agg = rel.groupby(["src", "dst"])["amount"].agg(["sum", "count"]).reset_index()
    agg["cp"] = np.where(agg["src"] == acct, agg["dst"], agg["src"])
    n_all = agg["cp"].nunique()
    keep = agg.groupby("cp")["sum"].sum().nlargest(max_nbrs).index
    agg = agg[agg["cp"].isin(keep)]
    ids = set(keep) | {acct}
    among = tx[tx["src"].isin(ids) & tx["dst"].isin(ids) & (tx["src"] != acct) & (tx["dst"] != acct)]
    agg2 = among.groupby(["src", "dst"])["amount"].agg(["sum", "count"]).reset_index()
    edges = [(r.src, r.dst, r["sum"], r["count"]) for _, r in pd.concat([agg, agg2]).iterrows()]
    return node_info(res, ids), edges, n_all > max_nbrs


def risk_legend():
    st.caption("節點顏色：紅＝高風險、橘＝中風險、灰＝低風險；黑框為調查對象；線條粗細代表金額；可拖曳、滾輪縮放、滑鼠停留看明細。")


def availability_for(run: dict) -> dict:
    acc_like = run["results"].reset_index()[["account_id"]]
    for c_ in dl.ACCOUNT_ATTRS:
        if c_ in run["results"]:
            acc_like[c_] = run["results"][c_].to_numpy()
    return rule_availability(run["tx"], acc_like)


# ---------------------------------------------------------------------------
# 頁面：總覽
# ---------------------------------------------------------------------------
def page_overview():
    run = current_run()
    res, s, m = run["results"], run["summary"], run["metrics"]
    st.title("總覽：全查成果")
    st.caption(f"資料集：{dl.DATASETS.get(s['name'], s['name'])}　期間：{s['date_start']} ~ {s['date_end']}")

    c = st.columns(5)
    c[0].metric("帳戶數（全查）", f"{s['n_accounts']:,}")
    c[1].metric("交易筆數（全查）", f"{s['n_transactions']:,}")
    c[2].metric("高風險帳戶", f"{(res['risk_level'] == '高').sum():,}")
    c[3].metric("命中任一紅旗規則", f"{(res['rule_hits'] > 0).sum():,}")
    c[4].metric("時間一致資金循環", f"{len(run['cycles']):,} 個")

    ev = eval_for(current_key()) if "uploaded_run" not in st.session_state else None
    if ev and ev.get("temporal"):
        t = ev["temporal"]["summary"]["100"]
        ai, rl = t["ai"], t["rules"]
        n_b = len(ev["temporal"]["batches"])
        st.success(
            f"**逐月持續稽核模擬（最接近實務的評估）**：每月只覆核風險最高的 100 個帳戶，{n_b} 個月共找出 "
            f"**{ai['unique_mules']} 個人頭帳戶**（規則只找到 {rl['unique_mules']} 個）；其中 {ai['caught_before_alert']} 個"
            f"比警方通報**平均早 {ai['lead_days_mean']:.0f} 天**，可攔阻被害款項 **{money(ai['prevented_amount'])}**"
            f"（規則 {money(rl['prevented_amount'])}）。詳見「模型評估」頁。", icon="📈")

    if m:
        st.subheader("同樣的人力，能找到多少人頭帳戶？")
        how = ("AI 分數為交叉驗證的樣本外分數（同一詐騙集團不會同時出現在訓練與測試）。" if m.get("folds")
               else "AI 分數由既有模型直接評分，模型訓練時從未看過這份資料。")
        tl = m.get("train_label", {})
        if tl.get("kind") == "alerted":
            how += (f"模型只用「已被通報警示」的 {tl['n_train_positive']} 個帳戶訓練（銀行實際只知道這些），"
                    f"但以全部 {tl['n_true_positive']} 個真實人頭帳戶評估。")
        st.caption("依風險由高到低逐一覆核帳戶，累計找到的人頭帳戶數。" + how)
        y = res["label"].fillna(0).to_numpy()
        n = len(y)
        kmax = int(min(n, max(4 * y.sum(), 500)))
        ks = np.unique(np.linspace(1, kmax, 300).astype(int))
        ai = np.cumsum(y[np.argsort(-res["risk_score"].to_numpy(), kind="stable")])[ks - 1]
        rl = np.cumsum(y[np.argsort(-rule_rank_score(res), kind="stable")])[ks - 1]
        rnd = ks * y.mean()
        fig = go.Figure()
        for name, vals, col in (("AI 模型（FlowAudit）", ai, C_AI), ("傳統規則", rl, C_RULE), ("隨機抽樣", rnd, C_RAND)):
            fig.add_trace(go.Scatter(x=ks, y=vals, name=name, mode="lines", line=dict(color=col, width=2),
                                     hovertemplate=f"{name}<br>覆核 %{{x:,}} 個帳戶<br>找到 %{{y:,.0f}} 個<extra></extra>"))
        fig.update_layout(hovermode="x unified")
        fig.update_xaxes(title="人工覆核帳戶數")
        fig.update_yaxes(title="找到的人頭帳戶數")
        st.plotly_chart(style_fig(fig, 400), use_container_width=True)

        k_show = next((r for r in m["model"]["at_k"] if r["k"] == 200), m["model"]["at_k"][0])
        r_show = next((r for r in m["rules"]["at_k"] if r["k"] == k_show["k"]), m["rules"]["at_k"][0])
        c = st.columns(4)
        c[0].metric(f"覆核前 {k_show['k']} 名命中率：AI", f"{k_show['precision']:.0%}")
        c[1].metric(f"覆核前 {k_show['k']} 名命中率：規則", f"{r_show['precision']:.0%}")
        c[2].metric("隨機抽樣命中率", f"{m['positive_rate']:.1%}")
        if m.get("unreported"):
            u = m["unreported"]
            kpi(c[3], "期間內未被通報的人頭帳戶", f"{u['n']} 個", f"其中 AI 找出 {u['found_in_top_n_pos']} 個")
        else:
            c[3].metric("抽樣要達到同樣檢出數需覆核", f"{k_show.get('random_accounts_needed', 0):,} 個帳戶")

        col1, col2 = st.columns(2)
        with col1:
            st.subheader("整體表現")
            tbl = pd.DataFrame({
                "方法": ["AI 模型", "傳統規則", "隨機抽樣"],
                "PR-AUC": [m["model"]["pr_auc"], m["rules"]["pr_auc"], m["random_sampling"]["pr_auc"]],
                "ROC-AUC": [m["model"]["roc_auc"], m["rules"]["roc_auc"], 0.5],
            })
            st.dataframe(tbl.style.format({"PR-AUC": "{:.3f}", "ROC-AUC": "{:.3f}"}), hide_index=True, use_container_width=True)
            if m.get("external_validation"):
                st.markdown("**外部驗證**：模型不重新訓練，直接套用到從未見過的資料")
                ev_ = pd.DataFrame([{"資料集": dl.DATASETS.get(k, "另一份仿真資料（不同隨機種子）" if k == "tw_sim_seed2" else k),
                                     "AI PR-AUC": v["model_pr_auc"], "規則 PR-AUC": v["rules_pr_auc"], "隨機基準": v["positive_rate"]}
                                    for k, v in m["external_validation"].items()])
                st.dataframe(ev_.style.format({c_: "{:.3f}" for c_ in ev_.columns[1:]}), hide_index=True, use_container_width=True)
        with col2:
            st.subheader("各紅旗規則命中情形")
            avail = availability_for(run)
            rows = []
            for code, name in RULES.items():
                h = res[f"{code}_hit"].astype(bool)
                rows.append({"規則": f"{code} {name}", "命中帳戶": int(h.sum()),
                             "其中為人頭帳戶": (f"{res.loc[h, 'label'].mean():.0%}" if h.sum() else "-") if avail[code] else "資料無此欄位"})
            st.dataframe(pd.DataFrame(rows), hide_index=True, use_container_width=True)
            st.caption("單一規則誤報很多（真實銀行的規則告警也常超過九成是誤報）；AI 綜合 50 多項特徵後大幅降低誤報。")
        st.info("本展示資料為合成資料。台灣情境仿真資料依公開的詐騙金流態樣與銀行限額設計，人頭帳戶比例刻意高於真實情況；"
                "實際導入時須以機構自有資料重新訓練與驗證。", icon="ℹ️")
    else:
        st.info("此資料沒有標籤，因此只顯示風險評分結果，不計算成效指標。")

    st.subheader("風險分數分布")
    fig = go.Figure()
    for lv in ["低", "中", "高"]:
        sub = res[res["risk_level"] == lv]
        fig.add_trace(go.Histogram(x=sub["risk_score"], name=f"{lv}風險（{len(sub):,}）", marker_color=C_RISK[lv],
                                   xbins=dict(start=0, end=1.0001, size=0.02), marker_line_width=0))
    fig.update_layout(barmode="stack", bargap=0.05)
    fig.update_yaxes(title="帳戶數", type="log")
    fig.update_xaxes(title="AI 風險分數")
    st.plotly_chart(style_fig(fig, 280), use_container_width=True)


# ---------------------------------------------------------------------------
# 頁面：風險帳戶清單
# ---------------------------------------------------------------------------
def page_list():
    run = current_run()
    res = run["results"]
    st.title("高風險帳戶清單")
    c = st.columns([1, 2, 1])
    levels = c[0].multiselect("風險等級", ["高", "中", "低"], default=["高", "中"])
    rule_sel = c[1].multiselect("命中規則（任一）", [f"{k} {v}" for k, v in RULES.items()])
    only_ai = c[2].checkbox("只看「規則全未命中、AI 判定可疑」", help="規則抓不到的新型態，最能展現 AI 價值")
    show_truth = is_sim(res) and st.checkbox("顯示仿真真實角色（僅供驗證，實務上不會有）")

    df = res[res["risk_level"].isin(levels)]
    if rule_sel:
        mask = np.zeros(len(df), dtype=bool)
        for r in rule_sel:
            mask |= df[f"{r[:2]}_hit"].to_numpy(dtype=bool)
        df = df[mask]
    if only_ai:
        df = df[df["rule_hits"] == 0]
    df = df.sort_values("risk_rank")
    st.caption(f"共 {len(df):,} 個帳戶；點選一列即可帶到「帳戶調查」頁。")
    cols = ["risk_rank", "account_id", "risk_score", "risk_level"]
    if "customer_type" in df.columns:
        cols.append("customer_type")
    cols += ["rule_list", "top_reasons", "n_in", "amt_in", "n_out", "amt_out"]
    show = df.reset_index()[cols]
    if show_truth:
        show["仿真真實角色"] = df["role"].to_numpy()
    elif res["label"].notna().any() and not is_sim(res):
        show["已知標籤"] = df["label"].map({1: "人頭帳戶", 0: "正常"}).to_numpy()
    sel = st.dataframe(
        show, hide_index=True, use_container_width=True, height=520, on_select="rerun", selection_mode="single-row",
        column_config={
            "risk_rank": st.column_config.NumberColumn("排名", format="%d"),
            "account_id": "帳戶", "customer_type": "類型",
            "risk_score": st.column_config.ProgressColumn("AI 風險分數", min_value=0, max_value=1, format="%.3f"),
            "risk_level": "等級", "rule_list": "命中規則", "top_reasons": "AI 主要判斷依據",
            "n_in": "匯入筆數", "amt_in": st.column_config.NumberColumn("匯入金額", format="%.0f"),
            "n_out": "匯出筆數", "amt_out": st.column_config.NumberColumn("匯出金額", format="%.0f"),
        },
    )
    if sel and sel.selection.rows:
        acct = show.iloc[sel.selection.rows[0]]["account_id"]
        st.session_state["investigate"] = acct
        st.success(f"已選取 {acct}，請切換到「帳戶調查」頁面查看細節。")

    c = st.columns(3)
    c[0].download_button("下載風險清單 CSV", show.to_csv(index=False).encode("utf-8-sig"), "risk_list.csv", "text/csv")
    k = c[1].number_input("本月覆核名額（工作底稿帳戶數）", 50, 2000, CFG["report"]["top_k_review"], step=50)
    n_min, quota = CFG["report"]["cycle_fallback_min"], CFG["report"].get("rule_quota", 0.0)
    c[2].download_button(
        "下載稽核工作底稿（Excel）",
        build_workpaper(res, run["tx"], run["metrics"], run["summary"], top_k=int(k), fallback_min=n_min, rule_quota=quota),
        "FlowAudit_稽核工作底稿.xlsx",
        "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
    )

    ai_ids, rule_ids = review_lists(res, int(k), quota)
    if len(rule_ids):
        rl = res.loc[rule_ids]
        with st.expander(f"規則名單（新手法保險）：{int(k)} 個覆核名額中，AI 名單 {len(ai_ids)} 個、"
                         f"規則分數最高但不在 AI 名單的 {len(rl)} 個"):
            st.caption("AI 只學得到見過的詐騙手法。保留少量名額給規則，新手法出現時仍有機會被發現；"
                       "平常幾乎不影響成效（見「模型評估」的新手法測試）。比例在 config.yaml 的 rule_quota 調整。")
            rshow = rl.reset_index()[["account_id", "rule_hits", "rule_list", "risk_rank", "risk_score"]
                                     + [c_ for c_ in ("customer_type", "amt_in", "amt_out") if c_ in rl.columns]]
            if show_truth:
                rshow["仿真真實角色"] = rl["role"].to_numpy()
            st.dataframe(rshow, hide_index=True, use_container_width=True, column_config={
                "account_id": "帳戶", "rule_hits": st.column_config.NumberColumn("命中規則數", format="%d"),
                "rule_list": "命中規則", "risk_rank": st.column_config.NumberColumn("AI 排名", format="%d"),
                "risk_score": st.column_config.NumberColumn("AI 風險分數", format="%.3f"), "customer_type": "類型",
                "amt_in": st.column_config.NumberColumn("匯入金額", format="%.0f"),
                "amt_out": st.column_config.NumberColumn("匯出金額", format="%.0f")})

    if "R4_cycles" in res.columns:
        fb = cycle_fallback(res, len(ai_ids), n_min)
        fb = fb[~fb.index.isin(rule_ids)]
        with st.expander(f"規則保底名單：覆核名單以外、參與資金循環 ≥ {n_min} 次的帳戶（{len(fb)} 個，另案專案查核）"):
            st.caption("循環交易帳戶多為公司戶、很少被通報警示，AI 缺少可學習的樣本；R4 規則抓得到它們，但只循環一次的誤報很多。"
                       "提高循環次數門檻後另列專案查核，補上 AI 的盲點。門檻在 config.yaml 的 cycle_fallback_min 調整；"
                       "各門檻的名單大小與命中情形見「模型評估」頁。")
            fcols = ["account_id", "risk_rank", "risk_score", "R4_cycles"] + \
                    [c_ for c_ in ("customer_type", "rule_list", "R4_evidence", "amt_in", "amt_out") if c_ in fb.columns]
            fshow = fb.reset_index()[fcols]
            if show_truth:
                fshow["仿真真實角色"] = fb["role"].to_numpy()
            st.dataframe(fshow, hide_index=True, use_container_width=True, column_config={
                "account_id": "帳戶", "risk_rank": st.column_config.NumberColumn("AI 排名", format="%d"),
                "risk_score": st.column_config.NumberColumn("AI 風險分數", format="%.3f"),
                "R4_cycles": st.column_config.NumberColumn("參與資金循環次數", format="%d"), "customer_type": "類型",
                "rule_list": "命中規則", "R4_evidence": "循環證據",
                "amt_in": st.column_config.NumberColumn("匯入金額", format="%.0f"),
                "amt_out": st.column_config.NumberColumn("匯出金額", format="%.0f")})


# ---------------------------------------------------------------------------
# 頁面：帳戶調查
# ---------------------------------------------------------------------------
def page_investigate():
    run = current_run()
    res, tx, shap_df, X = run["results"], run["tx"], run["shap"], run["X"]
    st.title("帳戶調查")
    ranked = res.sort_values("risk_rank").index.tolist()
    # 預設展示：前 30 名中命中規則最多、交易最多的帳戶（最能說明系統功能）
    top30 = res.loc[ranked[:30]]
    showcase = top30.sort_values(["rule_hits", "n_in"], ascending=False).index[0] if len(top30) else ranked[0]
    default = st.session_state.get("investigate", showcase)
    acct = st.selectbox("選擇帳戶（依風險排序，可輸入搜尋）", ranked, index=ranked.index(default) if default in ranked else 0)
    r = res.loc[acct]

    c = st.columns(5)
    c[0].metric("AI 風險分數", f"{r['risk_score']:.3f}")
    c[1].metric("風險等級", r["risk_level"])
    c[2].metric("全體排名", f"{int(r['risk_rank']):,} / {len(res):,}")
    c[3].metric("命中紅旗規則", f"{int(r['rule_hits'])} 項")
    c[4].metric("所屬疑似集團", f"G{int(r['group_id'])}（{int(r['group_size'])} 戶）" if r["group_id"] > 0 else "無")
    prof = []
    if "customer_type" in r.index and pd.notna(r.get("customer_type")):
        prof.append(f"帳戶類型：{r['customer_type']}")
    if "open_date" in r.index and pd.notna(r.get("open_date")):
        prof.append(f"開戶日：{pd.Timestamp(r['open_date']):%Y-%m-%d}")
    if "dormant_days_before" in r.index and pd.notna(r.get("dormant_days_before")):
        prof.append(f"觀察期前未往來：{int(r['dormant_days_before'])} 天")
    if X.at[acct, "R7_device_accounts"] > 1:
        prof.append(f"登入裝置共用帳戶數：{int(X.at[acct, 'R7_device_accounts'])}")
    if prof:
        st.caption("　｜　".join(prof))
    if is_sim(res):
        truth = str(r["role"]) + (f"（{r['scheme']}集團 {r['fraud_group']}）" if r.get("fraud_group") else "")
        alert = f"，警示日 {pd.Timestamp(r['alert_date']):%Y-%m-%d}" if pd.notna(r.get("alert_date")) else ""
        st.caption(f"仿真真實角色（僅供驗證，實務上不會知道）：{truth}{alert}")

    left, right = st.columns([1, 1])
    with left:
        st.subheader("AI 判斷依據（TreeSHAP）")
        s = shap_df.loc[acct].drop("_bias")
        top = s.reindex(s.abs().sort_values(ascending=False).index[:10])[::-1]
        names = [f"{FEATURE_NAMES_ZH.get(f, f)} = {X.at[acct, f]:,.2f}" for f in top.index]
        fig = go.Figure(go.Bar(
            x=top.values, y=names, orientation="h",
            marker_color=[C_RISK["高"] if v > 0 else C_AI for v in top.values],
            hovertemplate="%{y}<br>對風險的影響：%{x:+.2f}<extra></extra>",
        ))
        fig.update_xaxes(title="對風險的影響（← 降低｜提高 →）")
        st.plotly_chart(style_fig(fig, 420), use_container_width=True)
        st.caption("紅色：推高風險的因素；藍色：降低風險的因素。")

        st.subheader("命中的紅旗規則")
        if r["rule_hits"] == 0:
            st.info("未命中任何固定規則，本帳戶由 AI 依整體行為型態判定。")
        for code, name in RULES.items():
            if r[f"{code}_hit"]:
                with st.expander(f"⚠️ {code} {name}　（{RULE_REFS[code]}）", expanded=True):
                    st.write(RULE_DESCRIPTIONS[code])
                    st.markdown(f"**證據：** {r[f'{code}_evidence']}")

    with right:
        st.subheader("資金往來網路")
        nodes, edges, truncated = ego_graph(run, acct)
        components.html(network_html(nodes, edges, center=acct), height=540)
        risk_legend()
        if truncated:
            st.caption("往來對象較多，僅顯示金額最大的 40 個；現金存提不畫在網路圖上。")

    st.subheader("交易時間軸")
    t = tx[(tx["src"] == acct) | (tx["dst"] == acct)].copy()
    t["方向"] = np.where(t["dst"] == acct, "匯入", "匯出")
    t["對手帳戶"] = np.where(t["dst"] == acct, t["src"], t["dst"])
    t["對手帳戶"] = t["對手帳戶"].replace(CASH, "（現金）")
    fig = go.Figure()
    for d, col, sign in (("匯入", C_AI, 1), ("匯出", C_RULE, -1)):
        sub = t[t["方向"] == d]
        fig.add_trace(go.Bar(x=sub["date"], y=sign * sub["amount"], name=d, marker_color=col,
                             customdata=np.stack([sub["對手帳戶"], sub["amount"]], axis=1) if len(sub) else None,
                             hovertemplate=f"%{{x|%Y-%m-%d %H:%M}}<br>{d} %{{customdata[1]:,.0f}} 元<br>對手：%{{customdata[0]}}<extra></extra>"))
    fig.update_layout(barmode="relative", bargap=0.2)
    fig.update_yaxes(title="金額（匯入為正、匯出為負，元）")
    st.plotly_chart(style_fig(fig, 300), use_container_width=True)
    with st.expander(f"交易明細（{len(t)} 筆）"):
        cols = ["tx_id", "date", "方向", "對手帳戶", "amount"] + [c_ for c_ in ("channel", "device_id") if c_ in t.columns]
        st.dataframe(t[cols].rename(columns={"tx_id": "交易序號", "date": "時間", "amount": "金額", "channel": "管道",
                                             "device_id": "登入裝置"}), hide_index=True, use_container_width=True)

    st.subheader("自動產生可疑交易分析報告")
    provider = _llm_provider(CFG["report"]["llm"])
    c = st.columns([2, 1])
    use_llm = c[0].toggle("使用 LLM 撰寫（需設定 ANTHROPIC_API_KEY 或 OPENAI_API_KEY）", value=provider is not None,
                          disabled=provider is None, help="未設定金鑰時使用模板產生，完全離線也能運作。")
    if c[1].button("產生報告", type="primary", use_container_width=True):
        with st.spinner("撰寫報告中 …"):
            facts = collect_facts(acct, res, shap_df, X, tx, CFG["data"]["currency_label"])
            md, how = generate_report(facts, CFG["report"]["llm"], force_template=not use_llm)
        st.session_state["report"] = (acct, md, how)
    if st.session_state.get("report") and st.session_state["report"][0] == acct:
        _, md, how = st.session_state["report"]
        st.caption(f"產生方式：{ {'template': '模板', 'anthropic': 'Claude API', 'openai': 'OpenAI API'}.get(how, how) }")
        with st.container(border=True):
            st.markdown(md)
        c = st.columns(2)
        c[0].download_button("下載 Markdown", md.encode("utf-8"), f"報告_{acct}.md", "text/markdown")
        c[1].download_button("下載 Word", markdown_to_docx(md), f"報告_{acct}.docx",
                             "application/vnd.openxmlformats-officedocument.wordprocessingml.document")


# ---------------------------------------------------------------------------
# 頁面：疑似集團分析
# ---------------------------------------------------------------------------
def page_groups():
    run = current_run()
    res, tx = run["results"], run["tx"]
    st.title("疑似集團分析")
    st.caption("取出所有中、高風險帳戶及其彼此之間的資金往來，再以 Louvain 社群偵測切出往來緊密的群組，"
               "協助稽核人員以「集團」為單位擴大查核，而不是一個帳戶一個帳戶地看。")
    m = res[res["group_id"] > 0]
    if m.empty:
        st.info("沒有偵測到疑似集團。")
        return
    has_label = res["label"].notna().any()
    agg = dict(帳戶數=("risk_score", "size"), 高風險帳戶=("risk_level", lambda s: (s == "高").sum()),
               平均風險分數=("risk_score", "mean"), 命中規則帳戶=("rule_hits", lambda s: (s > 0).sum()))
    if has_label:
        agg["真實人頭帳戶比例"] = ("label", "mean")
    g = m.groupby("group_id").agg(**agg).sort_values(["高風險帳戶", "平均風險分數"], ascending=False)
    ids_all = set(m.index)
    inner = tx[tx["src"].isin(ids_all) & tx["dst"].isin(ids_all)].copy()
    inner["g1"] = inner["src"].map(res["group_id"])
    inner["g2"] = inner["dst"].map(res["group_id"])
    g["群內資金往來"] = inner[inner["g1"] == inner["g2"]].groupby("g1")["amount"].sum().reindex(g.index).fillna(0)
    g.index = [f"G{i}" for i in g.index]
    c = st.columns(3)
    c[0].metric("疑似集團數", f"{len(g):,}")
    c[1].metric("涵蓋可疑帳戶", f"{len(m):,}")
    c[2].metric("最大集團規模", f"{int(g['帳戶數'].max())} 戶")
    ccfg = {"平均風險分數": st.column_config.NumberColumn(format="%.3f"),
            "群內資金往來": st.column_config.NumberColumn(format="%.0f")}
    if has_label:
        ccfg["真實人頭帳戶比例"] = st.column_config.ProgressColumn(min_value=0, max_value=1, format="%.2f")
    sel = st.dataframe(g.reset_index().rename(columns={"index": "集團"}), hide_index=True, use_container_width=True,
                       height=300, on_select="rerun", selection_mode="single-row", column_config=ccfg)
    gid = g.index[sel.selection.rows[0]] if sel and sel.selection.rows else g.index[0]
    members = res[res["group_id"] == int(gid[1:])].sort_values("risk_rank")
    st.subheader(f"集團 {gid}：{len(members)} 個可疑帳戶，其中 {int((members['risk_level'] == '高').sum())} 個高風險")
    left, right = st.columns([3, 2])
    with left:
        ids = set(members.index[:80])
        sub = tx[tx["src"].isin(ids) & tx["dst"].isin(ids)]
        ag = sub.groupby(["src", "dst"])["amount"].agg(["sum", "count"]).reset_index()
        components.html(network_html(node_info(res, ids), [(r.src, r.dst, r["sum"], r["count"]) for _, r in ag.iterrows()]), height=540)
        risk_legend()
    with right:
        cols = ["account_id", "risk_score", "risk_level", "rule_list"]
        st.dataframe(members.reset_index()[cols], hide_index=True, use_container_width=True, height=470,
                     column_config={"account_id": "帳戶", "risk_score": st.column_config.ProgressColumn("AI 風險", min_value=0, max_value=1, format="%.3f"),
                                    "risk_level": "等級", "rule_list": "命中規則"})
        st.download_button("下載此集團帳戶清單", members.reset_index()[cols + ["top_reasons"]].to_csv(index=False).encode("utf-8-sig"),
                           f"group_{gid}.csv", "text/csv")


# ---------------------------------------------------------------------------
# 頁面：模型評估
# ---------------------------------------------------------------------------
def page_evaluation():
    st.title("模型評估")
    key = current_key()
    if "uploaded_run" in st.session_state:
        st.info("目前檢視的是上傳資料；評估結果以左側選擇的展示資料集為準。")
    ev = eval_for(key)
    if not ev:
        st.warning(f"「{dl.DATASETS.get(key, key)}」尚未執行完整評估。")
        if st.button("立即執行評估（約 3～5 分鐘）", type="primary"):
            from flowaudit.evaluation import run_evaluation
            box = st.status("評估中 …", expanded=True)
            run_evaluation(key, CFG, log=lambda msg: box.write(msg))
            box.update(label="完成", state="complete")
            st.cache_data.clear()
            st.rerun()
        st.code(f"python -m flowaudit.evaluation --dataset {key}")
        return

    tabs = st.tabs(["逐月持續稽核模擬", "成本效益試算", "多模型比較", "消融實驗", "誤判分析", "穩健性測試", "新手法測試",
                    "規則保底名單"])

    # --- 逐月模擬
    with tabs[0]:
        if not ev.get("temporal"):
            st.info("這份資料沒有交易時間或警示日期，無法進行逐月模擬（台灣情境仿真資料才有）。")
        else:
            tp = ev["temporal"]
            st.markdown("每月 1 日執行一次全查：**只用當時已有的交易**、**只知道當時已被警示的帳戶**，"
                        "對尚未被警示的帳戶排序並覆核前 K 名。被確認的人頭帳戶視為當天凍結，之後流入的被害款項即可攔阻。")
            k = st.radio("每月覆核帳戶數 K", ["50", "100", "200"], index=1, horizontal=True)
            s = tp["summary"][k]
            n_b = len(tp["batches"])
            c = st.columns(4)
            kpi(c[0], f"AI：{n_b} 個月找出的人頭帳戶", f"{s['ai']['unique_mules']} 個",
                f"規則 {s['rules']['unique_mules']} 個｜隨機抽樣約 {s['random']['unique_mules']:.0f} 個")
            kpi(c[1], "比警方通報更早發現", f"{s['ai']['caught_before_alert']} 個",
                f"平均早 {s['ai']['lead_days_mean']:.0f} 天" if s["ai"]["lead_days_mean"] else None)
            kpi(c[2], "期間內從未被通報、由系統找出", f"{s['ai']['never_alerted']} 個", "警方尚未發現的人頭帳戶")
            kpi(c[3], "可攔阻的被害款項", money(s["ai"]["prevented_amount"]), f"規則 {money(s['rules']['prevented_amount'])}")
            rows = tp["batches"]
            fig = go.Figure()
            for who, name, col in (("ai", "AI 模型", C_AI), ("rules", "傳統規則", C_RULE)):
                fig.add_trace(go.Bar(x=[r["batch"] for r in rows], y=[r["by_k"][k][who]["tp"] for r in rows], name=name,
                                     marker_color=col, hovertemplate=f"{name}<br>%{{x}}：找到 %{{y}} 個<extra></extra>"))
            fig.add_trace(go.Bar(x=[r["batch"] for r in rows], y=[r["by_k"][k]["random"]["tp"] for r in rows], name="隨機抽樣",
                                 marker_color=C_RAND, hovertemplate="隨機抽樣<br>%{x}：期望找到 %{y:.1f} 個<extra></extra>"))
            fig.update_layout(barmode="group", bargap=0.25, bargroupgap=0.05)
            fig.update_yaxes(title="找到的人頭帳戶數")
            fig.update_xaxes(title="批次日（每月 1 日）", type="category")
            st.markdown(f"**每月覆核 {k} 個帳戶，找到幾個尚未被通報的人頭帳戶**")
            st.plotly_chart(style_fig(fig, 360), use_container_width=True)
            tbl = pd.DataFrame([{
                "批次日": r["batch"], "已知警示帳戶": r["n_known_alerts"], "尚未被發現的人頭": r["hidden_mules"],
                "AI 找到": r["by_k"][k]["ai"]["tp"], "規則找到": r["by_k"][k]["rules"]["tp"],
                "AI 可攔阻金額": r["by_k"][k]["ai"]["prevented_amount"], "排序方式": r["mode"]} for r in rows])
            st.dataframe(tbl, hide_index=True, use_container_width=True,
                         column_config={"AI 可攔阻金額": st.column_config.NumberColumn(format="%.0f")})
            st.caption("前兩個月已知警示帳戶很少，AI 與規則差不多（已知警示帳戶不到 5 個時自動改用規則排序）；"
                       "警示資料累積後 AI 明顯超越規則。導入初期的空窗可用下方的「冷啟動」補上。"
                       "同一帳戶被找到後即凍結，不會在之後月份重複計算。")
            hy = s.get("hybrid")
            if hy:
                st.caption(f"我們也測試過「規則＋AI 加權」的雙層排序：{n_b} 個月找出 {hy['unique_mules']} 個，"
                           f"{'低於' if hy['unique_mules'] < s['ai']['unique_mules'] else '接近'} AI 單獨排序，"
                           "因此系統以 AI 分數排序、規則作為說明證據。")
            setting = tp.get("settings", {})
            options = [("ai", "AI（只用自家警示資料）"), ("rules", "傳統規則"),
                       ("dual", f"雙名單（保留 {setting.get('rule_quota', 0.1):.0%} 名額給規則）"),
                       ("cold", "冷啟動：參考資料與自家資料分布相同"),
                       ("cold_shift", "冷啟動：參考資料分布不同（較保守）")]
            if any(w in s for w, _ in options[2:]):
                st.markdown(f"**解決方案比較**（每月覆核 {k} 個帳戶，{n_b} 個月合計）")
                st.dataframe(pd.DataFrame([{
                    "排序方式": name, "找到人頭帳戶": s[w]["unique_mules"], "比警方通報更早": s[w]["caught_before_alert"],
                    "可攔阻被害款項": s[w]["prevented_amount"]} for w, name in options if w in s]),
                    hide_index=True, use_container_width=True,
                    column_config={"可攔阻被害款項": st.column_config.NumberColumn(format="%.0f")})
                st.caption(f"**冷啟動**：自家已知警示帳戶不到 {setting.get('switch_after', 50)} 個時，先用外部參考資料（代表同業或主管機關分享、"
                           "已完成調查的資料）訓練的模型，之後改用自家模型。導入初期正是最能攔阻被害款項的時候，警示資料卻最少，"
                           "冷啟動補上這段空窗。參考資料與自家資料都是同一個仿真器產生，實際效果會比表中低；"
                           "「分布不同」一列改用人頭帳戶行為差異很大的壓力測試資料當參考，是較保守的估計。"
                           "**雙名單**是新詐騙手法出現時的保險，說明見「新手法測試」。")

    # --- 成本效益
    with tabs[1]:
        if not ev.get("temporal"):
            st.info("需要逐月模擬結果。")
        else:
            st.markdown("以逐月模擬結果試算。人力假設可自行調整。")
            c = st.columns(3)
            k2 = c[0].selectbox("每月覆核帳戶數", ["50", "100", "200"], index=1)
            minutes = c[1].slider("每件覆核所需時間（分鐘）", 10, 120, 30, step=5)
            wage = c[2].slider("稽核人員時薪（元，假設值）", 300, 2000, 800, step=50)
            s = ev["temporal"]["summary"][k2]
            n_rev = s["reviews"]
            hours = n_rev * minutes / 60
            cost = hours * wage
            rows = []
            for who, name in (("ai", "AI 模型"), ("rules", "傳統規則")):
                x = s[who]
                rows.append({"方法": name, "覆核帳戶數": n_rev, "人工時數": hours, "人力成本（元）": cost,
                             "找到人頭帳戶": x["unique_mules"],
                             "每找到一個的成本（元）": cost / x["unique_mules"] if x["unique_mules"] else None,
                             "可攔阻被害款項（元）": x["prevented_amount"],
                             "攔阻金額／人力成本": x["prevented_amount"] / cost if cost else None})
            rnd = s["random"]["unique_mules"]
            rows.append({"方法": "隨機抽樣", "覆核帳戶數": n_rev, "人工時數": hours, "人力成本（元）": cost,
                         "找到人頭帳戶": round(rnd, 1), "每找到一個的成本（元）": cost / rnd if rnd else None,
                         "可攔阻被害款項（元）": np.nan, "攔阻金額／人力成本": np.nan})
            df = pd.DataFrame(rows)
            st.dataframe(df, hide_index=True, use_container_width=True, column_config={
                "人工時數": st.column_config.NumberColumn(format="%.0f"),
                "人力成本（元）": st.column_config.NumberColumn(format="%.0f"),
                "每找到一個的成本（元）": st.column_config.NumberColumn(format="%.0f"),
                "可攔阻被害款項（元）": st.column_config.NumberColumn(format="%.0f"),
                "攔阻金額／人力成本": st.column_config.NumberColumn(format="%.1f 倍")})
            ai = rows[0]
            st.success(f"每月覆核 {k2} 個帳戶、每件 {minutes} 分鐘，{len(ev['temporal']['batches'])} 個月約需 {hours:,.0f} 小時"
                       f"（{money(cost)}），AI 可找出 {ai['找到人頭帳戶']} 個人頭帳戶、攔阻 {money(ai['可攔阻被害款項（元）'])} 被害款項，"
                       f"約為人力成本的 {ai['攔阻金額／人力成本']:.0f} 倍。")
            st.caption("攔阻金額＝被系統找出後，原本還會流入該人頭帳戶的被害人匯款（仿真資料計算）。未計入避免的商譽損失與裁罰。")

    # --- 多模型比較
    with tabs[2]:
        ms = ev["models"]
        st.markdown(f"同一套{'集團層級' if ev.get('group_cv') else ''} 5 折交叉驗證切分"
                    f"{'（同一詐騙集團不會同時出現在訓練與測試）' if ev.get('group_cv') else ''}；橫線為 bootstrap 95% 信賴區間。")
        names = list(ms.keys())
        ap = [ms[n]["pr_auc"] for n in names]
        lo = [ms[n]["pr_auc"] - ms[n]["pr_auc_ci"][0] for n in names]
        hi = [ms[n]["pr_auc_ci"][1] - ms[n]["pr_auc"] for n in names]
        colors = [C_RULE if "規則計分" in n else C_AI if "FlowAudit" in n else "#9a9892" for n in names]
        fig = go.Figure(go.Bar(x=ap, y=names, orientation="h", marker_color=colors,
                               error_x=dict(type="data", symmetric=False, array=hi, arrayminus=lo, color="#52514e", thickness=1.2),
                               hovertemplate="%{y}<br>PR-AUC %{x:.3f}<extra></extra>"))
        fig.update_xaxes(title="PR-AUC（越高越好；隨機約等於人頭帳戶比例）", range=[0, 1.1])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 100 + 42 * len(names)), use_container_width=True)
        tbl = pd.DataFrame([{"模型": n, "PR-AUC": ms[n]["pr_auc"], "95% 信賴區間": f"{ms[n]['pr_auc_ci'][0]:.3f}～{ms[n]['pr_auc_ci'][1]:.3f}",
                             "各折標準差": ms[n]["pr_auc_fold_std"], "ROC-AUC": ms[n]["roc_auc"],
                             "前 100 名命中率": next(x["precision"] for x in ms[n]["at_k"] if x["k"] == 100)} for n in names])
        st.dataframe(tbl.style.format({"PR-AUC": "{:.3f}", "各折標準差": "{:.3f}", "ROC-AUC": "{:.3f}", "前 100 名命中率": "{:.0%}"}),
                     hide_index=True, use_container_width=True)
        st.caption("孤立森林不需要標籤，可在完全沒有警示資料時使用，但效果遠不如監督式模型；"
                   "隨機森林與 XGBoost 表現接近，系統採用 XGBoost，因為它能精確計算每個帳戶的 TreeSHAP 解釋且訓練快速。")
        if "規則加權（依警示資料學權重）" in ms and "XGBoost（只用 16 項規則指標）" in ms:
            w_, r16 = ms["規則加權（依警示資料學權重）"]["pr_auc"], ms["XGBoost（只用 16 項規則指標）"]["pr_auc"]
            st.caption(f"AI 的進步從哪裡來？只替 8 條規則的「是否命中」學權重，PR-AUC 為 {w_:.3f}，與現行規則差不多；"
                       f"改用規則背後的連續數值（例如快進快出比例、循環次數）而不切門檻，就達到 {r16:.3f}；"
                       f"再加上資金網路圖、時間管道等其餘特徵為 {ms['XGBoost（FlowAudit）']['pr_auc']:.3f}。"
                       "也就是說，大部分的進步來自「不把規則切成是／否」，其餘特徵提供額外但較小的幫助。")

    # --- 消融
    with tabs[3]:
        ab = ev["ablation"]
        base = ab["全部特徵"]["pr_auc"]
        groups = [g for g in ab if g != "全部特徵"]
        st.markdown(f"全部 {ab['全部特徵']['n_features']} 項特徵的 PR-AUC 為 **{base:.3f}**。"
                    "「只用此類」看單一類特徵本身有多少資訊；「拿掉此類」看它是否被其他特徵取代。")
        fig = go.Figure()
        fig.add_trace(go.Bar(y=groups, x=[ab[g]["only"] for g in groups], orientation="h", name="只用此類", marker_color=C_AI,
                             hovertemplate="%{y}：只用此類 PR-AUC %{x:.3f}<extra></extra>"))
        fig.add_trace(go.Bar(y=groups, x=[ab[g]["without"] for g in groups], orientation="h", name="拿掉此類", marker_color=C_RULE,
                             hovertemplate="%{y}：拿掉此類 PR-AUC %{x:.3f}<extra></extra>"))
        fig.add_vline(x=base, line_dash="dash", line_color="#52514e", annotation_text=f"全部特徵 {base:.3f}", annotation_position="top")
        fig.update_layout(barmode="group")
        fig.update_xaxes(title="PR-AUC", range=[0, 1.05])
        fig.update_yaxes(autorange="reversed")
        st.plotly_chart(style_fig(fig, 380), use_container_width=True)
        st.dataframe(pd.DataFrame([{"特徵類別": g, "特徵數": ab[g]["n_features"], "只用此類": ab[g]["only"],
                                    "拿掉此類": ab[g]["without"], "拿掉後變化": -ab[g]["drop"],
                                    "包含": "、".join(FEATURE_NAMES_ZH.get(c_, c_) for c_ in FEATURE_GROUPS[g][:6]) + ("…" if len(FEATURE_GROUPS[g]) > 6 else "")}
                                   for g in groups]).style.format({"只用此類": "{:.3f}", "拿掉此類": "{:.3f}", "拿掉後變化": "{:+.3f}"}),
                     hide_index=True, use_container_width=True)
        st.caption("任一類特徵拿掉後成效只小幅下降，代表各類特徵彼此互補、系統不依賴單一訊號，較不容易被詐騙集團針對性規避。")

    # --- 誤判
    with tabs[4]:
        er = ev["errors"]
        if er.get("fp_rate_by_role"):
            st.markdown(f"「AI 誤判」＝正常帳戶進入風險前 {er['k']} 名；「規則誤判」＝正常帳戶命中任一規則。")
            fr = pd.DataFrame([{"帳戶類型": k, "帳戶數": v["n"], "規則誤判率": v["rules"], "AI 誤判率": v["ai"]}
                               for k, v in er["fp_rate_by_role"].items()]).sort_values("規則誤判率", ascending=False)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=fr["帳戶類型"], x=fr["規則誤判率"], orientation="h", name="傳統規則", marker_color=C_RULE,
                                 hovertemplate="%{y}：規則誤判 %{x:.1%}<extra></extra>"))
            fig.add_trace(go.Bar(y=fr["帳戶類型"], x=fr["AI 誤判率"], orientation="h", name="AI 模型", marker_color=C_AI,
                                 hovertemplate="%{y}：AI 誤判 %{x:.1%}<extra></extra>"))
            fig.update_layout(barmode="group")
            st.markdown("**規則把大量正常帳戶當成可疑，AI 誤判少得多**")
            fig.update_xaxes(title="被誤判為可疑的比例", tickformat=".0%")
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_fig(fig, 80 + 30 * len(fr)), use_container_width=True)
            st.caption("團購主、代收班費、標會會頭、個人賣家、發薪公司等，行為與人頭帳戶相似但完全正常，是規則誤報的主要來源。")
        c = st.columns(2)
        if er.get("recall_by_scheme"):
            c[0].markdown(f"**各詐騙類型的檢出率**（前 {er['k']} 名）")
            c[0].dataframe(pd.DataFrame([{"類型": k, "檢出率": v} for k, v in er["recall_by_scheme"].items()])
                           .style.format({"檢出率": "{:.0%}"}), hide_index=True, use_container_width=True)
        if er.get("fn_by_source"):
            c[1].markdown("**漏掉的人頭帳戶來源**")
            c[1].dataframe(pd.DataFrame([{"帳戶來源": k or "（其他）", "數量": v} for k, v in er["fn_by_source"].items()]),
                           hide_index=True, use_container_width=True)
        if er.get("fn_by_scheme"):
            st.caption("循環交易帳戶多為公司戶、很少被通報警示，模型幾乎沒有可學習的樣本，是 AI 最難辨識的類型；"
                       "系統以「規則保底名單」補上（見最後一頁）。未來可加入發票、營業額等外部資料。")

    # --- 穩健性
    with tabs[5]:
        ls = ev.get("label_scarcity")
        rule_ap = ev["models"]["現行規則計分"]["pr_auc"]
        if ls:
            st.markdown("**警示資料很少時還有用嗎？** 隨機只保留一部分已警示帳戶訓練（其餘當作未知），仍以全部人頭帳戶評估；"
                        "每個比例抽 3 次取平均，陰影為最低～最高。")
            keys = sorted(ls, key=float)
            xs = [ls[k_]["n_train_positive"] for k_ in keys]
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=xs + xs[::-1], y=[ls[k_]["pr_auc_max"] for k_ in keys] + [ls[k_]["pr_auc_min"] for k_ in keys][::-1],
                                     fill="toself", fillcolor="rgba(42,120,214,0.15)", line=dict(width=0), hoverinfo="skip", showlegend=False))
            fig.add_trace(go.Scatter(x=xs, y=[ls[k_]["pr_auc"] for k_ in keys], mode="lines+markers", name="AI 模型",
                                     line=dict(color=C_AI, width=2),
                                     customdata=[f"{float(k_):.0%}" for k_ in keys],
                                     hovertemplate="已知警示 %{x} 個（%{customdata}）<br>PR-AUC %{y:.3f}<extra></extra>"))
            fig.add_hline(y=rule_ap, line_dash="dash", line_color=C_RULE, annotation_text=f"傳統規則 {rule_ap:.3f}（不需標籤）",
                          annotation_position="bottom right")
            fig.update_xaxes(title="訓練用的已警示帳戶數", type="log")
            fig.update_yaxes(title="PR-AUC", range=[0, 1.05])
            st.plotly_chart(style_fig(fig, 340), use_container_width=True)
            lo = ls[keys[0]]
            st.caption(f"只知道 {lo['n_train_positive']} 個警示帳戶（{float(keys[0]):.0%}）時，AI 的 PR-AUC 仍有 {lo['pr_auc']:.3f}、"
                       f"前 100 名命中率 {lo['p_at_100']:.0%}，高於傳統規則的 {rule_ap:.3f}。警示資料越多越好，但不需要大量標籤才能起步。")
        st_ = ev.get("stress")
        if st_:
            st.markdown("**壓力測試**：讓人頭帳戶更隱蔽、正常帳戶更像人頭，每個情境重新產生一份仿真資料。"
                        "「重新訓練」看方法本身能否適應；「不重新訓練」直接套用原本的模型，模擬詐騙手法改變但模型尚未更新。")
            names = list(st_)
            fig = go.Figure()
            fig.add_trace(go.Bar(y=names, x=[st_[n]["ai_pr_auc"] for n in names], orientation="h", name="AI（重新訓練）",
                                 marker_color=C_AI, hovertemplate="%{y}<br>AI 重新訓練 %{x:.3f}<extra></extra>"))
            fig.add_trace(go.Bar(y=names, x=[st_[n]["transfer_ai_pr_auc"] for n in names], orientation="h", name="AI（不重新訓練）",
                                 marker_color=C_AI, opacity=0.45, hovertemplate="%{y}<br>AI 不重新訓練 %{x:.3f}<extra></extra>"))
            fig.add_trace(go.Bar(y=names, x=[st_[n]["rules_pr_auc"] for n in names], orientation="h", name="傳統規則",
                                 marker_color=C_RULE, hovertemplate="%{y}<br>規則 %{x:.3f}<extra></extra>"))
            fig.update_layout(barmode="group")
            fig.update_xaxes(title="PR-AUC", range=[0, 1.05])
            fig.update_yaxes(autorange="reversed")
            st.plotly_chart(style_fig(fig, 120 + 70 * len(names)), use_container_width=True)
            st.dataframe(pd.DataFrame([{
                "情境": n, "人頭帳戶": v["n_mules"], "AI 重新訓練": v["ai_pr_auc"], "AI 不重新訓練": v["transfer_ai_pr_auc"],
                "傳統規則": v["rules_pr_auc"], "循環交易：AI 名單＋保底名單": f"{v['cycle']['ai_top_k'] + v['cycle']['fallback']}/{v['cycle']['n']}",
            } for n, v in st_.items()]).style.format({"AI 重新訓練": "{:.3f}", "AI 不重新訓練": "{:.3f}", "傳統規則": "{:.3f}"}),
                hide_index=True, use_container_width=True)
            st.caption("參數設定見 docs/仿真設計與參數依據.md。壓力測試仍是合成資料，只能說明「在這些假設下」的相對表現。")
        if not ls and not st_:
            st.info("尚未執行。請執行 python -m flowaudit.evaluation --stress")

    # --- 新手法測試
    with tabs[6]:
        un = ev.get("unseen")
        if not un:
            st.info("這份資料沒有詐騙類型資訊，或尚未執行評估（python -m flowaudit.evaluation）。")
        else:
            st.markdown("**AI 什麼時候會失效？** 監督式模型只學得到見過的手法。這裡在訓練時完全拿掉某一類詐騙的警示帳戶，"
                        f"模擬「新手法剛出現、還沒有人被通報」，看覆核前 {un['k']} 名能找到多少該類帳戶。")
            sch = un["schemes"]
            ks_ = [k_ for k_ in next(iter(sch.values()))["learning"]]
            fig = go.Figure()
            for i, (sc, v) in enumerate(sch.items()):
                fig.add_trace(go.Scatter(x=[int(k_) for k_ in ks_], y=[v["learning"][k_] for k_ in ks_], mode="lines+markers",
                                         name=sc, line=dict(color=C_AI, width=2, dash=["solid", "dash", "dot"][i % 3]),
                                         marker=dict(symbol=["circle", "square", "diamond"][i % 3], size=8),
                                         hovertemplate=f"{sc}<br>已知 %{{x}} 個該類警示帳戶<br>其餘同類帳戶找回 %{{y:.0%}}<extra></extra>"))
            fig.update_xaxes(title="訓練資料中該類手法的已警示帳戶數（0＝完全沒見過）", tickvals=[int(k_) for k_ in ks_])
            fig.update_yaxes(title="其餘同類帳戶找回比例", tickformat=".0%", range=[0, 1.05])
            st.plotly_chart(style_fig(fig, 320), use_container_width=True)
            st.dataframe(pd.DataFrame([{"詐騙類型": sc, "帳戶數": v["n"], "AI 見過時": v["recall_seen"],
                                        **{("完全沒見過" if k_ == "0" else f"只知道 {k_} 個"): v["learning"][k_] for k_ in ks_}}
                                       for sc, v in sch.items()]).style.format({c_: "{:.0%}" for c_ in
                                       ["AI 見過時"] + [("完全沒見過" if k_ == "0" else f"只知道 {k_} 個") for k_ in ks_]}),
                         hide_index=True, use_container_width=True)
            lo_, hi_ = min(v["learning"]["0"] for v in sch.values()), max(v["learning"]["0"] for v in sch.values())
            k_max = ks_[-1]
            st.caption(f"沒見過的手法，AI 只找回 {lo_:.0%}～{hi_:.0%}；只要累積 {k_max} 個該類警示帳戶，每月重新訓練後就提高到 "
                       f"{min(v['learning'][k_max] for v in sch.values()):.0%}～{max(v['learning'][k_max] for v in sch.values()):.0%}。"
                       "持續稽核（每月重新訓練）比一次性建模更重要。")
            st.markdown("**新手法出現前的保險：排序策略的取捨**（覆核名額相同）")
            strat = un["strategies"]
            scs = list(sch)
            st.dataframe(pd.DataFrame([{"排序策略": n, "全部手法都見過：找到人頭": v["hits_all_seen"],
                                        **{f"{sc}沒見過：該類找回": v["recall_unseen"][sc] for sc in scs}}
                                       for n, v in strat.items()]).style.format({f"{sc}沒見過：該類找回": "{:.0%}" for sc in scs}),
                         hide_index=True, use_container_width=True)
            st.caption("保留越多名額給規則，新手法出現時越保險，但平常找到的人頭帳戶會變少。這是風險胃納的選擇："
                       "系統預設「雙名單」保留少量名額給規則（config.yaml 的 rule_quota），平常幾乎沒有損失；"
                       "若主管機關或 165 發布新詐騙手法警訊，可暫時提高規則比重。")

    # --- 規則保底名單
    with tabs[7]:
        fbs = ev.get("fallback")
        if not fbs:
            st.info("尚未執行。請重新執行 python -m flowaudit.evaluation")
        else:
            n_min = CFG["report"]["cycle_fallback_min"]
            st.markdown(f"AI 名單（前 {next(iter(fbs.values()))['top_k']} 名）以外，**參與時間一致資金循環達 N 次**的帳戶另列專案查核。"
                        f"下表列出所有門檻 N 的結果（目前設定 N = {n_min}），不只挑一個好看的數字。"
                        + ("兩份資料由不同隨機種子產生，外部資料的模型從未見過。" if "tw_sim_seed2" in fbs else ""))
            cols_ = st.columns(len(fbs))
            for col, (name, fb) in zip(cols_, fbs.items()):
                title = "主要仿真資料" if name == "tw_sim" else "外部資料（另一個隨機種子）" if name == "tw_sim_seed2" else dl.DATASETS.get(name, name)
                col.markdown(f"**{title}**")
                if "n_cycle_mules" in fb:
                    col.caption(f"循環交易帳戶 {fb['n_cycle_mules']} 個，AI 名單內 {fb['cycle_in_ai_top_k']} 個")
                df_ = pd.DataFrame([{"門檻 N": r["min_cycles"], "名單帳戶數": r["n_queue"], "其中人頭": r["mules"],
                                     **({"其中循環交易": r["cycle_mules"]} if "cycle_mules" in r else {}),
                                     "命中率": r["precision"]} for r in fb["rows"]])
                col.dataframe(df_.style.format({"命中率": "{:.0%}"}).apply(
                    lambda row: ["font-weight: 700" if row["門檻 N"] == n_min else "" for _ in row], axis=1),
                    hide_index=True, use_container_width=True)
            st.caption("門檻越高名單越短、命中率越高，但太高會漏掉循環次數較少的集團。循環交易也可能是虛增營收等其他舞弊，"
                       "建議由稽核人員另案查核交易背後的商業實質。")
            sys_ = fbs.get("tw_sim", next(iter(fbs.values()))).get("system")
            if sys_ and sys_.get("rule_list_n"):
                st.markdown(f"**系統預設的完整流程**（主要仿真資料；覆核名額 {sys_['ai_list_n'] + sys_['rule_list_n']} 個，"
                            f"其中 {sys_['rule_quota']:.0%} 給規則名單；保底名單另案查核）")
                st.dataframe(pd.DataFrame([
                    {"名單": "AI 名單", "帳戶數": sys_["ai_list_n"], "其中人頭": sys_["ai_list_mules"], "其中循環交易": sys_["ai_list_cycle"]},
                    {"名單": "規則名單（新手法保險）", "帳戶數": sys_["rule_list_n"], "其中人頭": sys_["rule_list_mules"],
                     "其中循環交易": sys_["rule_list_cycle"]},
                    {"名單": f"規則保底名單（循環 ≥ {sys_['min_cycles']} 次）", "帳戶數": sys_["fallback_n"],
                     "其中人頭": sys_["fallback_mules"], "其中循環交易": sys_["fallback_cycle"]},
                ]), hide_index=True, use_container_width=True)
                st.caption(f"若同樣的覆核名額全部給 AI，可找到 {sys_['ai_only_mules']} 個人頭帳戶，但幾乎都不是循環交易；"
                           "保留少量名額給規則，就能把 AI 學不到的循環交易帳戶納入覆核。")


# ---------------------------------------------------------------------------
# 頁面：上傳新資料
# ---------------------------------------------------------------------------
def page_upload():
    st.title("上傳新資料進行全查")
    st.markdown(
        "上傳交易 CSV，系統會以目前資料集訓練好的模型，對**每一筆交易、每一個帳戶**執行規則檢核與 AI 評分。\n\n"
        "**必要欄位：** `src`（匯出帳戶）、`dst`（匯入帳戶）、`amount`（金額），以及 `step`（第幾天）或 `date`（日期時間）。\n\n"
        "**選填欄位：** `channel`（交易管道；現金存提請把對手帳戶填 `CASH`）、`device_id`（登入裝置）。\n\n"
        "**選填帳戶檔：** `account_id`, `label`（1＝已知人頭帳戶），可另含 `customer_type`、`open_date`、`dormant_days_before`。"
    )
    c = st.columns(2)
    f_tx = c[0].file_uploader("交易資料 CSV", type="csv")
    f_lb = c[1].file_uploader("帳戶／標籤 CSV（選填）", type="csv")
    sample = pd.DataFrame({"src": ["A1", "A2", "A3", "A9"], "dst": ["A9", "A9", "A1", "CASH"], "amount": [30000, 25000, 5000, 20000],
                           "date": ["2026-03-01 10:15", "2026-03-01 11:02", "2026-03-02 09:30", "2026-03-01 13:40"],
                           "channel": ["網路銀行", "行動銀行", "網路銀行", "ATM提款"], "device_id": ["D1", "D2", "D3", ""]})
    st.download_button("下載格式範例", sample.to_csv(index=False).encode("utf-8-sig"), "sample_format.csv", "text/csv")

    if f_tx and st.button("開始全查", type="primary"):
        try:
            ds = dl.load_generic_csv(f_tx, f_lb, CFG["data"]["base_date"], name=Path(f_tx.name).stem)
        except Exception as e:
            st.error(f"讀取失敗：{e}")
            return
        prog = st.status("全查進行中 …", expanded=True)
        run = analyze(ds, CFG, model=get_model(current_key()), log=lambda msg: prog.write(msg))
        prog.update(label=f"完成，耗時 {run['elapsed_sec']} 秒", state="complete")
        st.session_state["uploaded_run"] = run
        st.session_state.pop("investigate", None)
        st.success("已切換為上傳資料的分析結果，可到其他頁面檢視。")

    if st.session_state.get("uploaded_run"):
        if st.button("切換回展示資料集"):
            st.session_state.pop("uploaded_run")
            st.session_state.pop("investigate", None)
            st.rerun()


# ---------------------------------------------------------------------------
# 頁面：系統架構、資料與設定
# ---------------------------------------------------------------------------
def page_about():
    st.title("系統架構、資料與設定")
    st.markdown(
        """
```
交易資料（CSV／資料庫；含時間、管道、登入裝置、開戶資料時可發揮完整功能）
   │
   ├─① 規則引擎：8 項紅旗規則全查，每項對應調查局疑似洗錢態樣代碼或金管會管理辦法
   ├─② 特徵工程：交易行為、資金流速、時間與管道、資金網路圖、帳戶屬性與裝置，共 50 多項
   ├─③ XGBoost 風險模型：只用已警示帳戶訓練（貼近實務），集團層級交叉驗證
   ├─④ TreeSHAP：逐帳戶說明「為什麼可疑」
   ├─⑤ 疑似集團偵測：在可疑帳戶子圖上做 Louvain 社群偵測
   ├─⑥ 覆核名單：AI 名單＋規則名單（新手法保險）＋規則保底名單（資金循環，另案查核）
   ├─⑦ 報告生成：模板／LLM 撰寫可疑交易分析報告，匯出 Word 與 Excel 工作底稿
   └─⑧ 嚴謹評估：多模型比較、消融實驗、逐月持續稽核模擬（含冷啟動）、成本效益、誤判分析、
                 警示稀少、新手法測試、壓力測試
```

**為什麼要三份名單？** AI 只學得到見過、且有人被通報過的手法。沒見過的新手法與很少被通報的循環交易，
由規則名單與規則保底名單補上；新手法累積少數警示帳戶後，每月重新訓練的 AI 就能追上。
導入初期自家警示資料很少時，可先用外部參考資料訓練的模型（冷啟動）。
"""
    )
    avail = availability_for(current_run())
    st.subheader("紅旗規則與法規依據")
    st.dataframe(pd.DataFrame([{"代碼": k, "規則": v, "對應態樣／法規": RULE_REFS[k], "說明": RULE_DESCRIPTIONS[k],
                                "目前資料可完整運作": "是" if avail[k] else "否（缺欄位，改用替代判斷或不觸發）"}
                               for k, v in RULES.items()]), hide_index=True, use_container_width=True)
    st.subheader("台灣情境仿真資料")
    st.markdown(
        "- **正常帳戶**：上班族、學生、退休人士、自營業者、商家、公司、第二帳戶，以及容易被誤判的團購主、代收班費、"
        "標會會頭、個人賣家、房東、發薪公司、家人共用手機、記帳士代管公司網銀、休眠後領保險金。\n"
        "- **人頭帳戶**：來源為新開戶、久未往來的休眠帳戶、被出售的正常帳戶；詐騙類型有假投資、網購詐騙、假客服解除分期、循環交易。\n"
        "- **金流**：被害人 → 第一層人頭 → 第二層人頭 → 車手 ATM 分次提領（單次 2 萬元）或轉入虛擬貨幣交易所；"
        "被害人報案後帳戶警示凍結，集團換下一個人頭帳戶。\n"
        "- **限制**：合成資料，人頭帳戶比例約 1.4%（真實更低）；參數依公開資料設計，詳見 docs/仿真設計與參數依據.md。"
    )
    st.subheader("目前門檻設定（config.yaml）")
    st.json(CFG["rules"], expanded=False)
    st.markdown("**覆核名單與冷啟動設定**")
    st.json({"覆核名額": CFG["report"]["top_k_review"], "規則名單比例": CFG["report"].get("rule_quota", 0),
             "保底名單循環次數門檻": CFG["report"]["cycle_fallback_min"], "冷啟動": CFG["model"].get("cold_start", {})},
            expanded=False)
    st.subheader("模型特徵")
    st.dataframe(pd.DataFrame([{"類別": g, "特徵": c_, "說明": FEATURE_NAMES_ZH.get(c_, c_)} for g, cs in FEATURE_GROUPS.items() for c_ in cs]),
                 hide_index=True, use_container_width=True, height=320)


PAGES = {
    "總覽": page_overview,
    "高風險帳戶清單": page_list,
    "帳戶調查": page_investigate,
    "疑似集團分析": page_groups,
    "模型評估": page_evaluation,
    "上傳新資料全查": page_upload,
    "系統架構、資料與設定": page_about,
}

with st.sidebar:
    st.markdown("## 🔎 FlowAudit 金流偵探")
    st.caption("以資金流向圖與可解釋 AI 偵測人頭帳戶的持續稽核系統")
    page = st.radio("功能", list(PAGES), label_visibility="collapsed")
    st.divider()
    keys = ["tw_sim", "amlsim_fanin_cycle"]
    sel = st.selectbox("展示資料集", keys, index=keys.index(current_key()) if current_key() in keys else 0,
                       format_func=lambda k: dl.DATASETS[k])
    if sel != current_key():
        st.session_state["dataset"] = sel
        st.session_state.pop("uploaded_run", None)
        st.session_state.pop("investigate", None)
        st.rerun()
    if st.session_state.get("uploaded_run"):
        st.caption(f"目前檢視：上傳資料 {st.session_state['uploaded_run']['summary']['name']}")

PAGES[page]()
