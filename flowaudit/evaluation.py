"""嚴謹評估：回答評審最常問的「你怎麼知道模型真的有用？」

1. 多模型比較：現行規則計分、非監督孤立森林、邏輯斯迴歸、隨機森林、XGBoost，
   同一套集團層級交叉驗證切分，附 bootstrap 95% 信賴區間。
2. 消融實驗：逐一拿掉（或只用）某一類特徵，看成效掉多少 → 哪類特徵真正有貢獻。
3. 逐月持續稽核模擬（時間切分）：每月初只用「當時已有」的交易與「當時已知」的警示帳戶訓練，
   對尚未被警示的帳戶排序、覆核前 K 名，計算抓到幾個、平均提前幾天、可攔阻多少被害款項。
4. 成本效益：覆核人力時數與成本 vs. 攔阻金額。
5. 誤判分析：被誤判的是哪些正常帳戶、漏掉的是哪種人頭帳戶。
6. 警示資料稀少：銀行只知道一小部分警示帳戶時，模型還剩多少成效。
7. 新手法測試：訓練時拿掉某一類詐騙，AI 能否找到；新手法累積幾個警示帳戶後模型能學會；各排序策略的取捨。
8. 規則保底名單：AI 名單外、參與資金循環達門檻的帳戶，各門檻的名單大小與命中情形（含外部資料）。
9. 壓力測試（--stress）：人頭帳戶更隱蔽、轉出更慢、分散收款、困難正常樣本加倍時，AI 與規則各掉多少。
逐月模擬另比較「雙名單」與「冷啟動」（自家警示不足時先用參考資料訓練的模型）。

用法：python -m flowaudit.evaluation [--dataset tw_sim] [--quick] [--stress]
"""
from __future__ import annotations

import argparse
import json
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
from .features import FEATURE_GROUPS, RULE_FEATURES, build_features
from .model import (_params, cv_groups, cycle_fallback, dual_list, dump_json, hybrid_score, load_model, make_folds,
                    precision_recall_at_k, review_lists, rule_rank_score)
from .pipeline import OUT_DIR, analyze, load_config, load_run, training_labels
from .rules import RULES, run_rules


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
    lr = lambda: make_pipeline(StandardScaler(), LogisticRegression(class_weight="balanced", max_iter=3000))
    hits = rules.reindex(X.index)[[f"{c}_hit" for c in RULES]].astype(float)
    models = {
        "現行規則計分": None,
        # 銀行最容易做到的改良：用同一批警示帳戶替 8 條規則的命中結果學權重
        "規則加權（依警示資料學權重）": ("sup", lr, hits),
        "孤立森林（非監督）": ("unsup", lambda: IsolationForest(n_estimators=300, random_state=0, n_jobs=-1), X),
        "邏輯斯迴歸": ("sup", lr, X),
        "隨機森林": ("sup", lambda: RandomForestClassifier(n_estimators=300, min_samples_leaf=2, class_weight="balanced_subsample",
                                                        n_jobs=-1, random_state=0), X),
        # 只用規則的連續數值、不切門檻：拆解 AI 的進步有多少來自「保留規則數值」、多少來自其他特徵
        "XGBoost（只用 16 項規則指標）": ("sup", lambda: _xgb(cfg, y_train), X[RULE_FEATURES]),
        "XGBoost（FlowAudit）": ("sup", lambda: _xgb(cfg, y_train), X),
    }
    out, scores = {}, {}
    for name, spec in models.items():
        t0 = time.time()
        if spec is None:
            s = rule_rank_score(rules.reindex(X.index))
        else:
            kind, mk, data = spec
            s = oof_scores(mk, data, y_train, folds, unsupervised=(kind == "unsup"))
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
        log(f"  {g:<10} 拿掉後 {out[g]['without']:.3f}（{-out[g]['drop']:+.3f}）｜只用 {out[g]['only']:.3f}")
    return out


def label_scarcity(X, y_true, y_train, folds, cfg, log, fracs=(1.0, 0.5, 0.25, 0.1, 0.05), reps=3):
    """警示資料很少時還有用嗎？隨機只保留一部分已警示帳戶當訓練正樣本（其餘視為未知），
    仍以全部真實人頭帳戶評估；每個比例抽 reps 次取平均。"""
    known = np.flatnonzero(y_train.to_numpy() == 1)
    yv = np.asarray(y_true)
    n_pos = int(yv.sum())
    out = {}
    for frac in fracs:
        n = max(int(round(len(known) * frac)), 1)
        runs = []
        for rep in range(1 if frac == 1.0 else reps):
            keep = np.random.default_rng(100 + rep).choice(known, size=n, replace=False)
            yt = pd.Series(0, index=X.index)
            yt.iloc[keep] = 1
            s = oof_scores(lambda: _xgb(cfg, yt), X, yt, folds)
            at = precision_recall_at_k(yv, s, [100, n_pos])
            runs.append((average_precision_score(yv, s), at[0]["precision"], at[1]["precision"]))
        r = np.array(runs)
        out[f"{frac:g}"] = {"n_train_positive": n, "reps": len(runs), "pr_auc": float(r[:, 0].mean()),
                            "pr_auc_min": float(r[:, 0].min()), "pr_auc_max": float(r[:, 0].max()),
                            "p_at_100": float(r[:, 1].mean()), "p_at_n_pos": float(r[:, 2].mean())}
        log(f"  已知警示 {frac:>4.0%}（{n:>3} 個）：PR-AUC {r[:, 0].mean():.3f}"
            f"（{r[:, 0].min():.3f}～{r[:, 0].max():.3f}）｜前 100 名命中 {r[:, 1].mean():.0%}")
    return out


def fallback_eval(res: pd.DataFrame, top_k: int, ns=range(2, 11), rule_quota: float = 0.0, n_min: int = 5) -> dict:
    """規則保底名單在各種循環次數門檻下的名單大小與命中情形（列出全部門檻，不只挑一個）。

    另計算系統預設的完整流程：雙名單（AI 名單＋規則名單）再加上保底名單，三份名單各找到多少。
    """
    lab = res["label"].fillna(0)
    has_scheme = "scheme" in res.columns
    cyc = res["scheme"] == "循環交易" if has_scheme else pd.Series(False, index=res.index)
    out = {"top_k": top_k, "rows": []}
    if has_scheme:
        out["n_cycle_mules"] = int(cyc.sum())
        out["cycle_in_ai_top_k"] = int((cyc & (res["risk_rank"] <= top_k)).sum())
    for n in ns:
        q = cycle_fallback(res, top_k, n)
        row = {"min_cycles": n, "n_queue": len(q), "mules": int(lab[q.index].sum()),
               "precision": float(lab[q.index].mean()) if len(q) else 0.0}
        if has_scheme:
            row["cycle_mules"] = int(cyc[q.index].sum())
        out["rows"].append(row)
    ai_ids, rule_ids = review_lists(res, top_k, rule_quota)
    fb = cycle_fallback(res, len(ai_ids), n_min)
    fb = fb[~fb.index.isin(rule_ids)]
    ai_only = res.index[np.argsort(-res["risk_score"].to_numpy(), kind="stable")[:top_k]]
    out["system"] = {"rule_quota": rule_quota, "min_cycles": n_min, "ai_only_mules": int(lab[ai_only].sum()),
                     **{f"{name}_{what}": v for name, ids in (("ai_list", ai_ids), ("rule_list", rule_ids), ("fallback", fb.index))
                        for what, v in (("n", len(ids)), ("mules", int(lab[ids].sum())), ("cycle", int(cyc[ids].sum())))}}
    return out


def unseen_scheme(X, res, y_true, y_train, folds, cfg, log, ks_known=(0, 3, 10), reps=3):
    """新手法測試：訓練時拿掉某一類詐騙的已警示帳戶（只留 k 個，當作這種手法剛出現），
    看模型能否找到其餘同類帳戶。覆核名額＝人頭帳戶總數。

    同時比較幾種排序策略在「全部手法都見過」與「某手法沒見過」時的表現，讓稽核人員依風險胃納取捨：
    AI 單獨、雙名單（保留規則名額）、規則＋AI 加權、規則單獨。
    """
    yv = np.asarray(y_true)
    n_pos = int(yv.sum())
    rule = rule_rank_score(res)
    quota = cfg["report"].get("rule_quota", 0.1)
    strategies = {
        "AI 單獨": lambda s: np.argsort(-s, kind="stable")[:n_pos],
        f"雙名單（規則 {quota:.0%}）": lambda s: dual_list(s, rule, n_pos, quota),
        "雙名單（規則 20%）": lambda s: dual_list(s, rule, n_pos, 0.2),
        "規則＋AI 加權（AI 90%）": lambda s: np.argsort(-hybrid_score(s, rule, 0.9), kind="stable")[:n_pos],
        "規則＋AI 加權（AI 70%）": lambda s: np.argsort(-hybrid_score(s, rule, 0.7), kind="stable")[:n_pos],
        "規則單獨": lambda s: np.argsort(-rule, kind="stable")[:n_pos],
    }
    s_seen = res["risk_score"].to_numpy()
    out = {"k": n_pos, "strategies": {n: {"hits_all_seen": int(yv[f(s_seen)].sum())} for n, f in strategies.items()},
           "schemes": {}}
    y_tr = y_train.to_numpy()
    top_seen = np.argsort(-s_seen, kind="stable")[:n_pos]
    # 循環交易的已警示帳戶太少（見規則保底名單），不納入
    for sc in [s for s in res["scheme"].unique() if s and s != "循環交易"]:
        is_sc = (res["scheme"] == sc).to_numpy()
        known_sc = np.flatnonzero(is_sc & (y_tr == 1))
        rec = {"n": int(is_sc.sum()), "recall_seen": float(np.isin(np.flatnonzero(is_sc), top_seen).mean()), "learning": {}}
        for k in ks_known:
            vals, s0 = [], None
            for rep in range(reps if k else 1):
                keep = np.random.default_rng(rep).choice(known_sc, size=min(k, len(known_sc)), replace=False) if k else []
                yt = y_train.copy()
                yt.iloc[known_sc] = 0
                yt.iloc[keep] = 1
                s = oof_scores(lambda: _xgb(cfg["model"], yt), X, yt, folds)
                tgt = np.flatnonzero(is_sc & (yt.to_numpy() == 0))
                vals.append(float(np.isin(tgt, np.argsort(-s, kind="stable")[:n_pos]).mean()))
                if k == 0:
                    s0 = s
            rec["learning"][str(k)] = float(np.mean(vals))
            if k == 0:
                tgt = np.flatnonzero(is_sc)
                for n, f in strategies.items():
                    pick = f(s0)
                    out["strategies"][n].setdefault("recall_unseen", {})[sc] = float(np.isin(tgt, pick).mean())
                    out["strategies"][n].setdefault("hits_unseen", {})[sc] = int(yv[pick].sum())
        out["schemes"][sc] = rec
        log(f"  {sc}：見過 {rec['recall_seen']:.0%}｜沒見過 {rec['learning']['0']:.0%}｜"
            + "｜".join(f"已知 {k} 個 {rec['learning'][str(k)]:.0%}" for k in ks_known if k))
    return out


# 壓力測試情境：讓人頭帳戶更隱蔽、正常帳戶更像人頭（參數說明見 simulator.DEFAULT_CFG）
_HARD_NEG = ("n_landlord", "n_groupbuy", "n_collector", "n_family_mgr", "n_seller", "n_rosca", "n_reactivated")
_COVERT = {"mule_source_p": [0.15, 0.15, 0.70], "p_mule_ctrl": 0.2, "p_test_tx": 0.15}
_SLOW = {"fwd_fast_until": 0.2, "fwd_slow_until": 0.8}
_SPREAD = {"mule_cap": [2, 5]}
STRESS_SEED = 20260103  # 與主要資料、外部驗證資料都不同


def stress_scenarios() -> dict:
    from .simulator import DEFAULT_CFG
    hard = {k: DEFAULT_CFG[k] * 2 for k in _HARD_NEG}
    return {
        "基準（新隨機種子）": {},
        "隱蔽型人頭：多為出售帳戶、少共用裝置、少測試交易": _COVERT,
        "慢速轉出：多數隔 1～3 天才轉出": _SLOW,
        "困難正常樣本加倍": hard,
        "分散收款：每個人頭帳戶只收 2～5 名被害人": _SPREAD,
        "以上全部": {**_COVERT, **_SLOW, **hard, **_SPREAD},
    }


def stress_data_dir(over: dict) -> "Path":
    """壓力測試資料的快取位置：依情境參數命名，參數改了就會重新產生。"""
    import hashlib
    h = hashlib.md5(json.dumps({"seed": STRESS_SEED, **over}, sort_keys=True).encode()).hexdigest()[:10]
    return OUT_DIR / "tw_sim" / "stress" / h


def stress_test(cfg: dict, log, top_k: int | None = None) -> dict:
    """壓力測試：每個情境各產生一份仿真資料，
    (a) 在該資料上重新訓練（集團層級交叉驗證）→ 方法本身能否適應；
    (b) 直接套用主要資料訓練的模型、不重新訓練 → 詐騙手法改變時模型衰退多少。"""
    from .simulator import generate
    top_k = top_k or cfg["report"]["top_k_review"]
    n_min = cfg["report"]["cycle_fallback_min"]
    base_model = load_model(OUT_DIR / "tw_sim" / "model.json")
    out = {}
    for name, over in stress_scenarios().items():
        t0 = time.time()
        d = stress_data_dir(over)
        if not (d / "transactions.csv.gz").exists():
            generate(d, cfg={"seed": STRESS_SEED, **over}, verbose=False)
        ds = dl.load_twsim(d, name=f"stress_{d.name}")
        r = analyze(ds, cfg, log=lambda *_: None)
        m, res = r["metrics"], r["results"]
        n_pos = int(res["label"].sum())
        at = lambda who: next(a["precision"] for a in m[who]["at_k"] if a["k"] == n_pos)
        transfer = analyze(ds, cfg, model=base_model, log=lambda *_: None)["metrics"]
        pos = res[res["label"] == 1]
        top = pos["risk_rank"] <= n_pos
        cyc = res["scheme"] == "循環交易"
        fb = cycle_fallback(res, top_k, n_min)
        out[name] = {
            "n_accounts": int(len(res)), "n_mules": n_pos, "n_train_positive": m["train_label"]["n_train_positive"],
            "params": over, "positive_rate": m["positive_rate"],
            "ai_pr_auc": m["model"]["pr_auc"], "rules_pr_auc": m["rules"]["pr_auc"],
            "transfer_ai_pr_auc": transfer["model"]["pr_auc"], "ai_p_at_n_pos": at("model"), "rules_p_at_n_pos": at("rules"),
            "recall_by_scheme": top.groupby(pos["scheme"]).mean().round(3).to_dict(),
            "cycle": {"n": int(cyc.sum()), "ai_top_k": int((cyc & (res["risk_rank"] <= top_k)).sum()),
                      "fallback": int(cyc[fb.index].sum()), "fallback_queue": int(len(fb))},
        }
        o = out[name]
        log(f"  {name}：AI {o['ai_pr_auc']:.3f}｜不重新訓練 {o['transfer_ai_pr_auc']:.3f}｜規則 {o['rules_pr_auc']:.3f}"
            f"（{time.time() - t0:.0f}s）")
    return out


# ---------------------------------------------------------------------------
def _window(ds: dl.Dataset, T, cfg: dict):
    """只用 T 之前的交易重算規則與特徵（帳戶屬性照舊；真實角色欄位不進特徵）。"""
    acc_all = ds.accounts.set_index(ds.accounts["account_id"].astype(str))
    tx = ds.transactions[ds.transactions["date"] < T].copy()
    ids = pd.Index(pd.unique(pd.concat([tx["src"], tx["dst"]]))).difference(list(dl.EXTERNAL_IDS))
    acc = acc_all.drop(columns="account_id").reindex(ids).rename_axis("account_id").reset_index()
    rules, _ = run_rules(tx, acc, cfg["rules"])
    X, _, _ = build_features(tx, acc, rules)
    return acc.set_index("account_id").reindex(X.index), rules.reindex(X.index), X


def cold_start_references(cfg: dict) -> dict:
    """冷啟動參考資料：設定檔指定的外部資料（分布相同），以及壓力測試「以上全部」情境的資料（分布不同，較保守）。
    參考資料只取已被警示的帳戶當正樣本，與自家模型的訓練方式一致。"""
    from .simulator import generate
    name = cfg["model"].get("cold_start", {}).get("reference")
    if not name:
        return {}
    ref = dl.RAW_DIR / name
    if not (ref / "transactions.csv.gz").exists():
        generate(ref, cfg={"seed": 20260102}, verbose=False)
    over = stress_scenarios()["以上全部"]
    shift = stress_data_dir(over)
    if not (shift / "transactions.csv.gz").exists():
        generate(shift, cfg={"seed": STRESS_SEED, **over}, verbose=False)
    return {"cold": dl.load_twsim(ref, name=name), "cold_shift": dl.load_twsim(shift, name=f"stress_{shift.name}")}


def temporal_audit(ds: dl.Dataset, cfg: dict, log, ks=(50, 100, 200)):
    """逐月持續稽核模擬。

    每個批次日 T（每月 1 日）：只用 T 之前的交易、T 之前已被警示的帳戶訓練模型，
    對「尚未被警示、也尚未被本系統抓到」的帳戶排序，覆核前 K 名。
    被覆核確認的人頭帳戶視為當天凍結：之後的被害人匯款可被攔阻，且不會在下個月重複計算。

    比較的排序方式：ai（自家模型）、rules（現行規則）、hybrid（規則＋AI 加權）、
    dual（雙名單：保留 rule_quota 名額給規則）、cold／cold_shift（冷啟動：自家警示不足時改用參考資料訓練的模型）。
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
    refs = cold_start_references(cfg)
    switch_after = cfg["model"].get("cold_start", {}).get("switch_after", 50)
    quota = cfg["report"].get("rule_quota", 0.1)
    methods = ("ai", "rules", "hybrid", "dual") + tuple(refs)
    caught = {(k, m): set() for k in ks for m in methods}
    rows = []
    for T in batches:
        t0 = time.time()
        acc, rules, X = _window(ds, T, cfg)
        known = (acc["alert_date"] < T).to_numpy()
        y_known = pd.Series(known.astype(int), index=X.index)
        truth = acc["label"].fillna(0).astype(int).to_numpy()
        s_rule = rule_rank_score(rules)
        if y_known.sum() >= 5:
            skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=cfg["model"]["random_state"])
            s_ai = oof_scores(lambda: _xgb(cfg["model"], y_known.to_numpy()), X, y_known, list(skf.split(X, y_known)))
            mode = "AI"
        else:
            s_ai = s_rule.copy()
            mode = "規則（已知警示帳戶太少，模型尚無法訓練）"
        s_hyb = hybrid_score(s_ai, s_rule, cfg["model"].get("hybrid_weight_ai", 0.7))
        scores = {"ai": s_ai, "rules": s_rule, "hybrid": s_hyb, "dual": s_ai}
        for name, ref in refs.items():
            if known.sum() >= switch_after:
                scores[name] = s_ai
                continue
            racc, _, RX = _window(ref, T, cfg)
            ry = racc["alert_date"].notna().astype(int).to_numpy()
            m = _xgb(cfg["model"], ry)
            m.fit(RX, ry)
            scores[name] = m.predict_proba(X.reindex(columns=RX.columns).fillna(0))[:, 1]
        hidden = int(truth[~known].sum())
        rec = {"batch": f"{T:%Y-%m-%d}", "n_accounts": int(len(X)), "n_known_alerts": int(known.sum()),
               "hidden_mules": hidden, "mode": mode,
               "cold_mode": "參考資料模型" if refs and known.sum() < switch_after else "自家模型", "by_k": {}}
        for k in ks:
            rec["by_k"][str(k)] = {}
            for who in methods:
                done = caught[(k, who)]
                cand = ~known & ~X.index.isin(list(done))
                cid = X.index[cand]
                if who == "dual":
                    order = dual_list(s_ai[cand], s_rule[cand], k, quota)
                else:
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
        extra = "".join(f"、冷啟動{'（分布不同）' if n == 'cold_shift' else ''} {a[n]['tp']:>3}" for n in refs)
        log(f"  {T:%Y-%m-%d}｜已知警示 {rec['n_known_alerts']:>3}｜尚未發現的人頭 {hidden:>3}｜覆核前 100 名抓到："
            f"AI {a['ai']['tp']:>3}、規則 {a['rules']['tp']:>3}、雙層 {a['hybrid']['tp']:>3}、雙名單 {a['dual']['tp']:>3}"
            f"{extra}（{time.time() - t0:.0f}s）")
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
    return {"batches": rows, "summary": summary,
            "settings": {"rule_quota": quota, "switch_after": switch_after,
                         "cold_reference": cfg["model"].get("cold_start", {}).get("reference")}}


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
def run_evaluation(key: str, cfg: dict, quick: bool = False, stress: bool = False, log=print) -> dict:
    t_start = time.time()
    ds = dl.load_dataset(key, cfg["data"]["base_date"])
    prev_path = OUT_DIR / key / "evaluation.json"
    prev = json.loads(prev_path.read_text(encoding="utf-8")) if prev_path.exists() else {}
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

    if not quick:
        log("⑤ 警示資料稀少時的成效（隨機只保留部分已警示帳戶訓練）")
        out["label_scarcity"] = label_scarcity(X, y_true, y_train, folds, cfg["model"], log)
        if "scheme" in res.columns:
            log("⑥ 新手法測試（訓練時拿掉某一類詐騙，看能否找到）")
            out["unseen"] = unseen_scheme(X, res.reindex(X.index), y_true, y_train, folds, cfg, log)
    log("⑦ 規則保底名單（AI 名單外、參與資金循環達門檻的帳戶）")
    top_k, n_min = cfg["report"]["top_k_review"], cfg["report"]["cycle_fallback_min"]
    fb_args = dict(rule_quota=cfg["report"].get("rule_quota", 0.0), n_min=n_min)
    out["fallback"] = {key: fallback_eval(res, top_k, **fb_args)}
    alt = dl.RAW_DIR / "tw_sim_seed2"
    if key == "tw_sim" and (alt / "transactions.csv.gz").exists() and run["model_path"]:
        r2 = analyze(dl.load_twsim(alt, name="tw_sim_seed2"), cfg, model=load_model(run["model_path"]), log=lambda *_: None)
        out["fallback"]["tw_sim_seed2"] = fallback_eval(r2["results"], top_k, **fb_args)
    for name, fb in out["fallback"].items():
        row = next(r for r in fb["rows"] if r["min_cycles"] == n_min)
        extra = f"，其中循環交易 {row['cycle_mules']}/{fb['n_cycle_mules']}" if "cycle_mules" in row else ""
        log(f"  {name}：門檻 {n_min} 次 → 名單 {row['n_queue']} 個、人頭 {row['mules']} 個{extra}")
        s_ = fb["system"]
        log(f"    完整流程（AI {s_['ai_list_n']}＋規則 {s_['rule_list_n']}＋保底 {s_['fallback_n']}）："
            f"人頭 {s_['ai_list_mules']}＋{s_['rule_list_mules']}＋{s_['fallback_mules']}，"
            f"循環交易 {s_['ai_list_cycle']}＋{s_['rule_list_cycle']}＋{s_['fallback_cycle']}（只用 AI 前 {top_k} 名：人頭 {s_['ai_only_mules']}）")

    if key == "tw_sim" and stress:
        log("⑧ 壓力測試（每個情境重新產生仿真資料，約 15 分鐘）")
        out["stress"] = stress_test(cfg, log)
    elif prev.get("stress"):
        out["stress"] = prev["stress"]  # 沿用上次的壓力測試結果（加 --stress 重跑）
    out["elapsed_sec"] = round(time.time() - t_start, 1)
    dump_json(out, OUT_DIR / key / "evaluation.json")
    log(f"完成，耗時 {out['elapsed_sec']} 秒 → outputs/{key}/evaluation.json")
    return out


def main():
    ap = argparse.ArgumentParser(description="FlowAudit 嚴謹評估")
    ap.add_argument("--dataset", default=None, choices=list(dl.DATASETS))
    ap.add_argument("--quick", action="store_true", help="略過逐月模擬與警示稀少測試")
    ap.add_argument("--stress", action="store_true", help="重跑壓力測試（約 15 分鐘；未指定時沿用上次結果）")
    args = ap.parse_args()
    cfg = load_config()
    key = args.dataset or cfg["data"].get("default_dataset", "tw_sim")
    run_evaluation(key, cfg, args.quick, args.stress)


if __name__ == "__main__":
    main()
