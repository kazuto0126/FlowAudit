"""嚴謹評估：回答評審最常問的「你怎麼知道模型真的有用？」

1. 多模型比較：現行規則計分、非監督孤立森林、邏輯斯迴歸、隨機森林、XGBoost，
   同一套集團層級交叉驗證切分，附 bootstrap 95% 信賴區間。
2. 消融實驗：逐一拿掉（或只用）某一類特徵，看成效掉多少 → 哪類特徵真正有貢獻。
3. 逐月持續稽核模擬（時間切分）：每月初只用「當時已有」的交易與「當時已知」的警示帳戶訓練，
   對尚未被警示的帳戶排序、覆核前 K 名，計算抓到幾個、平均提前幾天、可攔阻多少被害款項。
4. 成本效益：覆核人力時數與成本 vs. 攔阻金額。
5. 誤判分析：被誤判的是哪些正常帳戶、漏掉的是哪種人頭帳戶。

用法：python -m flowaudit.evaluation [--dataset tw_sim] [--quick]
"""
from __future__ import annotations

import argparse
import time

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.ensemble import IsolationForest, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import data_loader as dl
from .features import FEATURE_GROUPS, build_features
from .model import _params, cv_groups, dump_json, hybrid_score, make_folds, precision_recall_at_k, rule_rank_score
from .pipeline import OUT_DIR, load_config, load_run, training_labels
from .rules import run_rules


# ---------------------------------------------------------------------------
def _xgb(cfg, y, n_estimators=None):
    p = _params(cfg, np.asarray(y))
    if n_estimators:
        p["n_estimators"] = n_estimators
    return xgb.XGBClassifier(**p)


def oof_scores(make_model, X, y_train, folds, unsupervised=False):
    s = np.zeros(len(X))
    for tr, te in folds:
        m = make_model()
        if unsupervised:
            m.fit(X.iloc[tr])
            s[te] = -m.score_samples(X.iloc[te])  # 越異常分數越高
        else:
            m.fit(X.iloc[tr], y_train.iloc[tr])
            s[te] = m.predict_proba(X.iloc[te])[:, 1]
    return s


def bootstrap_ci(y, s, n=500, seed=0):
    rng = np.random.default_rng(seed)
    y, s = np.asarray(y), np.asarray(s)
    vals = []
    for _ in range(n):
        i = rng.integers(0, len(y), len(y))
        if y[i].sum() == 0:
            continue
        vals.append(average_precision_score(y[i], s[i]))
    return float(np.percentile(vals, 2.5)), float(np.percentile(vals, 97.5))


def summarize(y, s, folds, ks=(100, 200)):
    y = np.asarray(y)
    fold_ap = [average_precision_score(y[te], s[te]) for _, te in folds if y[te].sum() > 0]
    lo, hi = bootstrap_ci(y, s)
    at = precision_recall_at_k(y, s, list(ks) + [int(y.sum())])
    return {
        "pr_auc": float(average_precision_score(y, s)), "pr_auc_ci": [lo, hi],
        "pr_auc_fold_std": float(np.std(fold_ap)), "roc_auc": float(roc_auc_score(y, s)),
        "at_k": at,
    }


# ---------------------------------------------------------------------------
def compare_models(X, y_true, y_train, folds, cfg, rules, log):
    models = {
        "現行規則計分": None,
        "孤立森林（非監督）": ("unsup", lambda: IsolationForest(n_estimators=300, random_state=0, n_jobs=-1)),
        "邏輯斯迴歸": ("sup", lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))),
        "隨機森林": ("sup", lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                                        n_jobs=-1, random_state=0)),
        "XGBoost（FlowAudit）": ("sup", lambda: _xgb(cfg, y_train)),
    }
    out, scores = {}, {}
    for name, spec in models.items():
        t0 = time.time()
        if spec is None:
            s = rule_rank_score(rules.reindex(X.index))
        else:
            kind, mk = spec
            s = oof_scores(mk, X, y_train, folds, unsupervised=(kind == "unsup"))
        scores[name] = s
        out[name] = summarize(y_true, s, folds)
        log(f"  {name:<16} PR-AUC {out[name]['pr_auc']:.3f} "
            f"[{out[name]['pr_auc_ci'][0]:.3f}, {out[name]['pr_auc_ci'][1]:.3f}]  ({time.time() - t0:.0f}s)")
    w = cfg.get("hybrid_weight_ai", 0.7)
    name = f"雙層：規則＋XGBoost（{w:.0%}／{1 - w:.0%}）"
    scores[name] = hybrid_score(scores["XGBoost（FlowAudit）"], scores["現行規則計分"], w)
    out[name] = summarize(y_true, scores[name], folds)
    log(f"  {name:<16} PR-AUC {out[name]['pr_auc']:.3f}")
    return out, scores


def ablation(X, y_true, y_train, folds, cfg, log):
    out = {}
    full = oof_scores(lambda: _xgb(cfg, y_train, 250), X, y_train, folds)
    base = float(average_precision_score(y_true, full))
    out["全部特徵"] = {"pr_auc": base, "n_features": X.shape[1]}
    for g, cols in FEATURE_GROUPS.items():
        cols = [c for c in cols if c in X.columns]
        if not cols:
            continue
        rest = [c for c in X.columns if c not in cols]
        s_wo = oof_scores(lambda: _xgb(cfg, y_train, 250), X[rest], y_train, folds)
        s_only = oof_scores(lambda: _xgb(cfg, y_train, 250), X[cols], y_train, folds)
        out[g] = {
            "n_features": len(cols),
            "without": float(average_precision_score(y_true, s_wo)),
            "only": float(average_precision_score(y_true, s_only)),
        }
        out[g]["drop"] = base - out[g]["without"]
        log(f"  {g:<10} 拿掉後 {out[g]['without']:.3f}（-{out[g]['drop']:.3f}）｜只用 {out[g]['only']:.3f}")
    return out


# ---------------------------------------------------------------------------
def temporal_audit(ds: dl.Dataset, cfg: dict, log, ks=(50, 100, 200)):
    """逐月持續稽核模擬。

    每個批次日 T（每月 1 日）：只用 T 之前的交易、T 之前已被警示的帳戶訓練模型，
    對「尚未被警示、也尚未被本系統抓到」的帳戶排序，覆核前 K 名。
    被覆核確認的人頭帳戶視為當天凍結：之後的被害人匯款可被攔阻，且不會在下個月重複計算。
    """
    tx_all, acc_all = ds.transactions, ds.accounts
    acc_all = acc_all.set_index(acc_all["account_id"].astype(str))
    start = tx_all["date"].min().normalize().replace(day=1)
    end = tx_all["date"].max()
    batches = []
    T = start + pd.DateOffset(months=1)
    while T <= end + pd.Timedelta(days=1):
        batches.append(T)
        T += pd.DateOffset(months=1)
    pattern = tx_all["pattern"] if "pattern" in tx_all.columns else pd.Series("", index=tx_all.index)
    victim_in = tx_all[pattern == "被害人匯入"]
    methods = ("ai", "rules", "hybrid")
    caught = {(k, m): set() for k in ks for m in methods}
    rows = []
    for T in batches:
        t0 = time.time()
        tx = tx_all[tx_all["date"] < T].copy()
        ids = pd.Index(pd.unique(pd.concat([tx["src"], tx["dst"]]))).difference(list(dl.EXTERNAL_IDS))
        acc = acc_all.drop(columns="account_id").reindex(ids).rename_axis("account_id").reset_index()
        rules, _ = run_rules(tx, acc, cfg["rules"])
        X, _, _ = build_features(tx, acc, rules)
        known = (acc.set_index("account_id")["alert_date"].reindex(X.index) < T).to_numpy()
        y_known = pd.Series(known.astype(int), index=X.index)
        truth = acc.set_index("account_id")["label"].reindex(X.index).fillna(0).astype(int).to_numpy()
        s_rule = rule_rank_score(rules.reindex(X.index))
        if y_known.sum() >= 5:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["model"]["random_state"])
            s_ai = oof_scores(lambda: _xgb(cfg["model"], y_known.to_numpy()), X, y_known, list(skf.split(X, y_known)))
            mode = "AI"
        else:
            s_ai = s_rule.copy()
            mode = "規則（已知警示帳戶太少，模型尚無法訓練）"
        s_hyb = hybrid_score(s_ai, s_rule, cfg["model"].get("hybrid_weight_ai", 0.7))
        scores = {"ai": s_ai, "rules": s_rule, "hybrid": s_hyb}
        hidden = int(truth[~known].sum())
        rec = {"batch": f"{T:%Y-%m-%d}", "n_accounts": int(len(X)), "n_known_alerts": int(known.sum()),
               "hidden_mules": hidden, "mode": mode, "by_k": {}}
        for k in ks:
            rec["by_k"][str(k)] = {}
            for who in methods:
                done = caught[(k, who)]
                cand = ~known & ~X.index.isin(list(done))
                cid = X.index[cand]
                order = np.argsort(-scores[who][cand], kind="stable")[:k]
                picked = cid[order]
                tp = [a for a in picked if acc_all.at[a, "label"] == 1]
                done.update(tp)
                al = acc_all.loc[tp, "alert_date"]
                lead = (al.dropna() - T).dt.days
                prevented = float(victim_in[victim_in["dst"].isin(tp) & (victim_in["date"] >= T)]["amount"].sum())
                rec["by_k"][str(k)][who] = {
                    "tp": len(tp), "precision": len(tp) / k, "hidden_available": int(truth[cand].sum()),
                    "caught_before_alert": int(len(lead)), "lead_days_sum": float(lead.sum()),
                    "never_alerted": int(al.isna().sum()), "prevented_amount": prevented,
                }
            rec["by_k"][str(k)]["random"] = {"tp": hidden * k / max(int((~known).sum()), 1)}
        a = rec["by_k"]["100"]
        log(f"  {T:%Y-%m-%d}｜已知警示 {rec['n_known_alerts']:>3}｜尚未發現的人頭 {hidden:>3}｜覆核前 100 名抓到："
            f"AI {a['ai']['tp']:>3}、規則 {a['rules']['tp']:>3}、雙層 {a['hybrid']['tp']:>3}（{time.time() - t0:.0f}s）")
        rows.append(rec)
    summary = {}
    n_b = len(rows)
    for k in ks:
        ks_ = str(k)
        summary[ks_] = {"reviews": k * n_b}
        for who in methods:
            g = [r["by_k"][ks_][who] for r in rows]
            n_before = sum(x["caught_before_alert"] for x in g)
            summary[ks_][who] = {
                "unique_mules": sum(x["tp"] for x in g), "precision": sum(x["tp"] for x in g) / (k * n_b),
                "caught_before_alert": n_before,
                "lead_days_mean": sum(x["lead_days_sum"] for x in g) / n_before if n_before else None,
                "never_alerted": sum(x["never_alerted"] for x in g),
                "prevented_amount": sum(x["prevented_amount"] for x in g),
            }
        summary[ks_]["random"] = {"unique_mules": sum(r["by_k"][ks_]["random"]["tp"] for r in rows)}
    summary["total_mules"] = int(acc_all["label"].sum())
    summary["victim_amount_total"] = float(victim_in["amount"].sum())
    return {"batches": rows, "summary": summary}


# ---------------------------------------------------------------------------
def error_analysis(res: pd.DataFrame):
    n_pos = int(res["label"].sum())
    top = res.sort_values("risk_rank").head(n_pos)
    fp = top[top["label"] == 0]
    fn = res[(res["label"] == 1) & ~res.index.isin(top.index)]
    tp = top[top["label"] == 1]
    out = {"k": n_pos, "fp_by_role": fp["role"].value_counts().to_dict() if "role" in res else {},
           "fn_by_scheme": fn["scheme"].value_counts().to_dict() if "scheme" in res else {},
           "fn_by_source": fn["mule_source"].value_counts().to_dict() if "mule_source" in res else {},
           "fn_by_layer": fn["mule_layer"].astype(str).value_counts().to_dict() if "mule_layer" in res else {}}
    if "scheme" in res:
        pos = res[res["label"] == 1]
        out["recall_by_scheme"] = (pos.index.isin(tp.index)).astype(float)
        out["recall_by_scheme"] = pos.assign(hit=pos.index.isin(tp.index)).groupby("scheme")["hit"].mean().to_dict()
    if "role" in res:
        neg = res[res["label"] == 0]
        # 各類正常帳戶被規則誤判的比例 vs. 被 AI 誤判（進入前 n_pos 名）的比例
        out["fp_rate_by_role"] = {
            r: {"n": int(len(g)), "rules": float((g["rule_hits"] > 0).mean()), "ai": float(g.index.isin(fp.index).mean())}
            for r, g in neg.groupby("role") if len(g) >= 20
        }
    return out


# ---------------------------------------------------------------------------
def run_evaluation(key: str, cfg: dict, quick: bool = False, log=print) -> dict:
    t_start = time.time()
    ds = dl.load_dataset(key, cfg["data"]["base_date"])
    run = load_run(key)
    X, res = run["X"], run["results"]
    y_true = res["label"].reindex(X.index).fillna(0).astype(int)
    y_train = training_labels(ds, X.index, cfg, y_true)
    groups = cv_groups(ds.accounts, X.index)
    folds = make_folds(y_train, cfg["model"], groups)
    out = {"dataset": key, "n_accounts": int(len(X)), "n_true_positive": int(y_true.sum()),
           "n_train_positive": int(y_train.sum()), "group_cv": groups is not None}

    log("① 多模型比較（集團層級 5 折交叉驗證，95% 信賴區間為 bootstrap）")
    rules_cols = [c for c in res.columns if c.startswith("R") and "_" in c] + ["rule_hits"]
    out["models"], _ = compare_models(X, y_true, y_train, folds, cfg["model"], res[rules_cols], log)

    log("② 消融實驗（XGBoost）")
    out["ablation"] = ablation(X, y_true, y_train, folds, cfg["model"], log)

    if ds.has_alert_dates and ds.has_time and not quick:
        log("③ 逐月持續稽核模擬（只用當時已有的資料與已知警示帳戶）")
        out["temporal"] = temporal_audit(ds, cfg, log)
    log("④ 誤判分析")
    out["errors"] = error_analysis(res)
    out["elapsed_sec"] = round(time.time() - t_start, 1)
    dump_json(out, OUT_DIR / key / "evaluation.json")
    log(f"完成，耗時 {out['elapsed_sec']} 秒 → outputs/{key}/evaluation.json")
    return out


def main():
    ap = argparse.ArgumentParser(description="FlowAudit 嚴謹評估")
    ap.add_argument("--dataset", default=None, choices=list(dl.DATASETS))
    ap.add_argument("--quick", action="store_true", help="略過逐月模擬")
    args = ap.parse_args()
    cfg = load_config()
    key = args.dataset or cfg["data"].get("default_dataset", "tw_sim")
    run_evaluation(key, cfg, args.quick)


if __name__ == "__main__":
    main()
