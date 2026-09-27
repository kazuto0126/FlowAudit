"""端對端流程：資料 → 規則引擎 → 特徵 → 模型 → 解釋 → 輸出。

用法
    python -m flowaudit.pipeline                         # 預設：AMLSim fan-in + cycle 樣本
    python -m flowaudit.pipeline --dataset amlsim_cycle
    python -m flowaudit.pipeline --amlworld HI-Small_Trans.csv   # Kaggle IBM AMLworld
"""
from __future__ import annotations

import argparse
import pickle
import time
from pathlib import Path

import networkx as nx
import numpy as np
import pandas as pd
import yaml

from . import data_loader as dl
from .features import build_features
from .model import (cross_validated_scores, cv_groups, dump_json, evaluate, fit_final, load_model,
                    rule_rank_score, save_model, score_new)
from .report import risk_level, top_reasons_text
from .rules import run_rules

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "outputs"


def load_config(path: Path | None = None) -> dict:
    return yaml.safe_load(open(path or ROOT / "config.yaml", encoding="utf-8"))


def detect_groups(tx: pd.DataFrame, res: pd.DataFrame, resolution: float = 2.0) -> pd.DataFrame:
    """疑似集團偵測：只取中、高風險帳戶與它們彼此之間的資金往來，再用 Louvain 切出緊密群組。

    回傳每個帳戶的 group_id（-1 表示不屬於任何集團）、group_size、group_high_risk（群內其他高風險帳戶數）。
    """
    sus = set(res.index[res["risk_level"] != "低"])
    out = pd.DataFrame({"group_id": -1, "group_size": 0, "group_high_risk": 0}, index=res.index)
    if len(sus) < 2:
        return out
    e = tx[tx["src"].isin(sus) & tx["dst"].isin(sus)].groupby(["src", "dst"]).size().reset_index()
    G = nx.Graph()
    G.add_nodes_from(sus)
    G.add_weighted_edges_from(e.itertuples(index=False, name=None))
    comms = [c for c in nx.community.louvain_communities(G, seed=42, resolution=resolution) if len(c) >= 2]
    comms.sort(key=len, reverse=True)
    high = res["risk_level"] == "高"
    for i, c in enumerate(comms, 1):
        c = list(c)
        out.loc[c, "group_id"] = i
        out.loc[c, "group_size"] = len(c)
        out.loc[c, "group_high_risk"] = int(high[c].sum()) - high[c].astype(int).to_numpy()
    return out


def _assemble(ds: dl.Dataset, rules: pd.DataFrame, X: pd.DataFrame, meta: pd.DataFrame,
              prob: np.ndarray, shap_df: pd.DataFrame) -> pd.DataFrame:
    res = pd.DataFrame(index=X.index)
    res["risk_score"] = prob
    res["risk_rank"] = res["risk_score"].rank(ascending=False, method="first").astype(int)
    res["risk_level"] = res["risk_score"].map(risk_level)
    lab = ds.accounts.set_index("account_id")["label"].reindex(X.index)
    res["label"] = lab
    res = res.join(rules.reindex(X.index))
    res = res.join(X[["n_in", "amt_in", "n_out", "amt_out", "n_src", "n_dst"]])
    res = res.join(meta)
    res = res.join(detect_groups(ds.transactions, res))
    res["top_reasons"] = top_reasons_text(shap_df)
    # 帳戶屬性（銀行已知）與仿真真實角色（僅供驗證，不進模型）
    acc = ds.accounts.set_index(ds.accounts["account_id"].astype(str)).reindex(X.index)
    for c in dl.ACCOUNT_ATTRS + dl.GROUND_TRUTH_COLS:
        if c in acc.columns:
            res[c] = acc[c]
    return res


def training_labels(ds: dl.Dataset, index: pd.Index, cfg: dict, y: pd.Series) -> pd.Series:
    """模型訓練用的標籤。

    真實世界中銀行只知道「已被通報警示」的帳戶，還沒被發現的人頭帳戶看起來和正常帳戶一樣。
    設定 model.train_label: alerted 且資料有警示日期時，只用已警示帳戶當正樣本訓練，
    但成效仍以全部真實人頭帳戶評估，才能看出模型能否找到「還沒被通報」的帳戶。
    """
    if cfg["model"].get("train_label", "label") != "alerted" or not ds.has_alert_dates:
        return y
    al = ds.accounts.set_index(ds.accounts["account_id"].astype(str))["alert_date"].reindex(index)
    return al.notna().astype(int)


def analyze(ds: dl.Dataset, cfg: dict, model=None, log=print) -> dict:
    """分析一個資料集。

    - 有標籤且未提供模型：以交叉驗證訓練並評估，另訓練一個全資料模型供新資料使用。
    - 提供模型：直接評分（新資料／無標籤資料）；若資料有標籤則同時計算成效。
    """
    t0 = time.time()
    tx, acc = ds.transactions, ds.accounts
    log(f"[1/4] 規則引擎全查 {len(tx):,} 筆交易、{len(acc):,} 個帳戶 …")
    rules, cycles = run_rules(tx, acc, cfg["rules"])
    log(f"[2/4] 計算行為特徵與資金流向圖特徵 …")
    X, meta, _ = build_features(tx, acc, rules)
    y = acc.set_index("account_id")["label"].reindex(X.index)

    metrics, folds, final = None, None, model
    n_pos = max(int(y.fillna(0).sum()), 1)
    ks = sorted({50, 100, 200, 500, n_pos} | ({1000} if len(X) > 5000 else set()))
    if model is None:
        if not ds.has_labels:
            raise ValueError("此資料集沒有標籤，請提供已訓練的模型")
        log(f"[3/4] {cfg['model']['n_folds']} 折交叉驗證訓練 XGBoost …")
        y = y.fillna(0).astype(int)
        y_train = training_labels(ds, X.index, cfg, y)
        if not y_train.equals(y):
            log(f"      訓練標籤：期間內已被通報警示的帳戶 {int(y_train.sum())} 個（銀行實際只知道這些）；"
                f"評估標籤：全部 {int(y.sum())} 個人頭帳戶")
        groups = cv_groups(acc, X.index)
        if groups is not None:
            log("      （集團層級切分：同一詐騙集團不會同時出現在訓練與測試）")
        prob, shap_df, folds = cross_validated_scores(X, y_train, cfg["model"], groups=groups)
        final = fit_final(X, y_train, cfg["model"])
    else:
        log("[3/4] 以既有模型評分 …")
        prob, shap_df = score_new(model, X)
    if ds.has_labels:
        yy = y.fillna(0).astype(int)
        metrics = evaluate(yy, prob, rule_rank_score(rules.reindex(X.index)), ks)
        if model is None:
            yt = training_labels(ds, X.index, cfg, yy)
            metrics["train_label"] = {"kind": "alerted" if not yt.equals(yy) else "label",
                                      "n_train_positive": int(yt.sum()), "n_true_positive": int(yy.sum())}
            unrep = (yy == 1) & (yt == 0)
            if unrep.any():
                order = np.argsort(-prob)
                topk = set(X.index[order[: int(yy.sum())]])
                metrics["unreported"] = {"n": int(unrep.sum()),
                                         "found_in_top_n_pos": int(sum(a in topk for a in X.index[unrep.to_numpy()]))}
        if folds:
            metrics["folds"] = folds
    log("[4/4] 彙整結果與可解釋性分析 …")
    results = _assemble(ds, rules, X, meta, prob, shap_df)
    return {
        "summary": ds.summary(), "results": results, "shap": shap_df, "X": X, "tx": tx,
        "cycles": cycles, "metrics": metrics, "model": final, "elapsed_sec": round(time.time() - t0, 1),
    }


def save_run(run: dict, name: str) -> Path:
    d = OUT_DIR / name
    d.mkdir(parents=True, exist_ok=True)
    for k in ("results", "shap", "X", "tx"):
        run[k].to_pickle(d / f"{k}.pkl")
    with open(d / "cycles.pkl", "wb") as fh:
        pickle.dump(run["cycles"], fh)
    dump_json(run["summary"] | {"elapsed_sec": run["elapsed_sec"]}, d / "summary.json")
    if run["metrics"]:
        dump_json(run["metrics"], d / "metrics.json")
    if run["model"] is not None:
        save_model(run["model"], d / "model.json")
    run["results"].sort_values("risk_rank").drop(
        columns=[c for c in run["results"].columns if c.endswith("_evidence")]
    ).to_csv(d / "risk_scores.csv", encoding="utf-8-sig")
    return d


def load_run(name: str) -> dict:
    d = OUT_DIR / name
    run = {k: pd.read_pickle(d / f"{k}.pkl") for k in ("results", "shap", "X", "tx")}
    with open(d / "cycles.pkl", "rb") as fh:
        run["cycles"] = pickle.load(fh)
    import json
    run["summary"] = json.loads((d / "summary.json").read_text(encoding="utf-8"))
    m = d / "metrics.json"
    run["metrics"] = json.loads(m.read_text(encoding="utf-8")) if m.exists() else None
    run["model_path"] = d / "model.json" if (d / "model.json").exists() else None
    return run


def main():
    ap = argparse.ArgumentParser(description="FlowAudit 金流偵探：人頭帳戶偵測流程")
    ap.add_argument("--dataset", default=None, choices=list(dl.DATASETS))
    ap.add_argument("--amlworld", help="IBM AMLworld CSV 路徑（例：HI-Small_Trans.csv）")
    ap.add_argument("--nrows", type=int, default=None, help="AMLworld 只讀前 N 筆（測試用）")
    ap.add_argument("--skip-external", action="store_true", help="略過外部資料集驗證")
    args = ap.parse_args()
    cfg = load_config()
    key = args.dataset or cfg["data"].get("default_dataset", "tw_sim")

    if args.amlworld:
        ds = dl.load_amlworld(args.amlworld, nrows=args.nrows)
    else:
        ds = dl.load_dataset(key, cfg["data"]["base_date"])
    print("資料：", ds.summary())
    run = analyze(ds, cfg)
    out = save_run(run, ds.name)
    m = run["metrics"]
    print(f"\n完成（{run['elapsed_sec']} 秒），輸出於 {out}")
    print(f"AI 模型   PR-AUC {m['model']['pr_auc']:.3f}  ROC-AUC {m['model']['roc_auc']:.3f}")
    print(f"傳統規則  PR-AUC {m['rules']['pr_auc']:.3f}  ROC-AUC {m['rules']['roc_auc']:.3f}")
    print(f"隨機抽樣  PR-AUC {m['random_sampling']['pr_auc']:.3f}")
    for r_m, r_r in zip(m["model"]["at_k"], m["rules"]["at_k"]):
        print(f"  覆核前 {r_m['k']:>5} 名：AI 命中 {r_m['precision']:.1%}｜規則 {r_r['precision']:.1%}｜抽樣 {m['positive_rate']:.1%}")

    # 外部驗證：用本資料訓練的模型，直接評分「從未見過」的資料
    if args.amlworld or args.skip_external:
        return
    ext = {}
    if key == "tw_sim":
        from .simulator import generate
        alt_dir = dl.RAW_DIR / "tw_sim_seed2"
        if not (alt_dir / "transactions.csv.gz").exists():
            print("\n產生外部驗證用的第二份仿真資料（不同隨機種子）…")
            generate(alt_dir, cfg={"seed": 20260102}, verbose=False)
        targets = {"tw_sim_seed2": lambda: dl.load_twsim(alt_dir, name="tw_sim_seed2")}
    else:
        targets = {k: (lambda k=k: dl.load_amlsim(k, cfg["data"]["base_date"])) for k in dl.AMLSIM_DATASETS if k != key}
    for name, loader in targets.items():
        print(f"\n外部驗證：{name}")
        r2 = analyze(loader(), cfg, model=run["model"], log=lambda *_: None)
        ext[name] = {
            "model_pr_auc": r2["metrics"]["model"]["pr_auc"], "model_roc_auc": r2["metrics"]["model"]["roc_auc"],
            "rules_pr_auc": r2["metrics"]["rules"]["pr_auc"], "positive_rate": r2["metrics"]["positive_rate"],
            "model_at_k": r2["metrics"]["model"]["at_k"], "rules_at_k": r2["metrics"]["rules"]["at_k"],
        }
        print(f"  AI PR-AUC {ext[name]['model_pr_auc']:.3f}｜規則 {ext[name]['rules_pr_auc']:.3f}｜基準 {ext[name]['positive_rate']:.3f}")
    m["external_validation"] = ext
    dump_json(m, out / "metrics.json")


if __name__ == "__main__":
    main()
