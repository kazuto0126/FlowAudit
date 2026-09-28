"""機器學習模型：XGBoost 帳戶風險評分 + TreeSHAP 可解釋性。

- 以 5 折交叉驗證產生「樣本外」風險分數（每個帳戶的分數都來自沒看過它的模型），
  儀表板與成效評估都使用這份分數，避免高估模型表現。
- 解釋值使用 XGBoost 內建的 TreeSHAP（pred_contribs），與 shap 套件的 TreeExplainer 演算法相同，
  可精確拆解「每個特徵把風險分數往上或往下推了多少」。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold, StratifiedKFold


def _params(cfg: dict, y: np.ndarray) -> dict:
    p = dict(cfg["xgb"])
    pos = max(y.sum(), 1)
    ratio = float((len(y) - pos) / pos)
    spw = p.pop("scale_pos_weight", "sqrt")
    # 正負樣本極度不平衡：以「負／正比例的平方根」加權，兼顧召回與排序品質
    spw = np.sqrt(ratio) if spw == "sqrt" else ratio if spw == "balanced" else float(spw)
    p.update(
        objective="binary:logistic",
        eval_metric="aucpr",
        tree_method="hist",
        random_state=cfg["random_state"],
        scale_pos_weight=float(spw),
        n_jobs=-1,
    )
    return p


def _contribs(model: xgb.XGBClassifier, X: pd.DataFrame) -> np.ndarray:
    return model.get_booster().predict(xgb.DMatrix(X), pred_contribs=True)  # 最後一欄為 bias


def make_folds(y: pd.Series, cfg: dict, groups: pd.Series | None = None):
    """交叉驗證切分。有集團資訊時使用「集團層級」切分：同一詐騙集團的帳戶不會同時出現在訓練與測試，
    避免模型只是記住某個集團的特徵而高估成效。"""
    if groups is not None:
        sgkf = StratifiedGroupKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["random_state"])
        return list(sgkf.split(np.zeros(len(y)), y, groups))
    skf = StratifiedKFold(n_splits=cfg["n_folds"], shuffle=True, random_state=cfg["random_state"])
    return list(skf.split(np.zeros(len(y)), y))


def cross_validated_scores(X: pd.DataFrame, y: pd.Series, cfg: dict, groups: pd.Series | None = None, folds_idx=None):
    """回傳 (樣本外機率, 樣本外 SHAP 值 DataFrame, 各折評估)"""
    folds_idx = folds_idx or make_folds(y, cfg, groups)
    prob = np.zeros(len(X))
    contrib = np.zeros((len(X), X.shape[1] + 1))
    folds = []
    for k, (tr, te) in enumerate(folds_idx):
        m = xgb.XGBClassifier(**_params(cfg, y.values[tr]))
        m.fit(X.iloc[tr], y.values[tr])
        prob[te] = m.predict_proba(X.iloc[te])[:, 1]
        contrib[te] = _contribs(m, X.iloc[te])
        folds.append(
            {
                "fold": k + 1,
                "roc_auc": float(roc_auc_score(y.values[te], prob[te])),
                "pr_auc": float(average_precision_score(y.values[te], prob[te])),
            }
        )
    shap_df = pd.DataFrame(contrib[:, :-1], index=X.index, columns=X.columns)
    shap_df["_bias"] = contrib[:, -1]
    return prob, shap_df, folds


def fit_final(X: pd.DataFrame, y: pd.Series, cfg: dict) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier(**_params(cfg, y.values))
    m.fit(X, y.values)
    return m


def score_new(model: xgb.XGBClassifier, X: pd.DataFrame):
    cols = model.get_booster().feature_names
    X = X.reindex(columns=cols).fillna(0)
    prob = model.predict_proba(X)[:, 1]
    c = _contribs(model, X)
    shap_df = pd.DataFrame(c[:, :-1], index=X.index, columns=cols)
    shap_df["_bias"] = c[:, -1]
    return prob, shap_df


def save_model(model: xgb.XGBClassifier, path: Path):
    model.save_model(str(path))


def load_model(path: Path) -> xgb.XGBClassifier:
    m = xgb.XGBClassifier()
    m.load_model(str(path))
    return m


# ---------------------------------------------------------------------------
# 成效評估：AI vs 規則 vs 傳統隨機抽樣
# ---------------------------------------------------------------------------
def precision_recall_at_k(y: np.ndarray, score: np.ndarray, ks) -> list[dict]:
    order = np.argsort(-score, kind="stable")
    ys = y[order]
    tot = max(y.sum(), 1)
    out = []
    for k in ks:
        k = int(min(k, len(y)))
        tp = int(ys[:k].sum())
        out.append({"k": k, "hits": tp, "precision": tp / k, "recall": tp / tot})
    return out


def evaluate(y: pd.Series, prob: np.ndarray, rule_score: np.ndarray, ks) -> dict:
    yv = y.values.astype(int)
    base = float(yv.mean())
    res = {
        "positive_rate": base,
        "model": {
            "roc_auc": float(roc_auc_score(yv, prob)),
            "pr_auc": float(average_precision_score(yv, prob)),
            "at_k": precision_recall_at_k(yv, prob, ks),
        },
        "rules": {
            "roc_auc": float(roc_auc_score(yv, rule_score)),
            "pr_auc": float(average_precision_score(yv, rule_score)),
            "at_k": precision_recall_at_k(yv, rule_score, ks),
        },
        "random_sampling": {
            "pr_auc": base,
            "at_k": [{"k": int(k), "hits": base * k, "precision": base, "recall": k / len(yv)} for k in ks],
        },
    }
    # 抽樣要達到與 AI 相同的檢出數，需要看多少帳戶
    for row in res["model"]["at_k"]:
        row["random_accounts_needed"] = int(np.ceil(row["hits"] / base)) if base > 0 else None
    return res


def rank_pct(s: np.ndarray) -> np.ndarray:
    return pd.Series(np.asarray(s)).rank(pct=True, method="average").to_numpy()


def hybrid_score(ai: np.ndarray, rule: np.ndarray, w_ai: float = 0.7) -> np.ndarray:
    """規則＋AI 雙層分數：兩者的百分位排名加權平均。AI 擅長已知型態，規則能補上 AI 沒見過的新型態。"""
    return w_ai * rank_pct(ai) + (1 - w_ai) * rank_pct(rule)


def rule_rank_score(rules: pd.DataFrame) -> np.ndarray:
    """規則基準線的排序分數：命中規則數為主，快進快出比例為次（僅用於排序，不含標籤）。"""
    return (rules["rule_hits"] + 0.5 * rules["R1_event_amount_share"] + 0.01 * np.log1p(rules["R4_cycles"])).to_numpy()


def dual_list(ai: np.ndarray, rule: np.ndarray, k: int, rule_quota: float) -> np.ndarray:
    """雙名單：k 個覆核名額中，(1 - rule_quota) 給 AI 分數最高者，其餘給「不在 AI 名單、規則分數最高」者。

    AI 只學得到見過的詐騙手法；保留少數名額給規則，是新手法出現時的保險。回傳位置索引（AI 名單在前）。
    """
    ai, rule = np.asarray(ai), np.asarray(rule)
    k = min(k, len(ai))
    k_ai = int(round(k * (1 - rule_quota)))
    o_ai = np.argsort(-ai, kind="stable")[:k_ai]
    rest = np.setdiff1d(np.arange(len(ai)), o_ai)
    return np.r_[o_ai, rest[np.argsort(-rule[rest], kind="stable")[: k - k_ai]]]


def review_lists(res: pd.DataFrame, top_k: int, rule_quota: float = 0.0) -> tuple[pd.Index, pd.Index]:
    """覆核名單：依 dual_list 分配 top_k 個名額，回傳 (AI 名單, 規則名單) 的帳戶代號。"""
    need = {"rule_hits", "R1_event_amount_share", "R4_cycles"}
    if rule_quota <= 0 or not need <= set(res.columns):
        rule_quota, rule = 0.0, np.zeros(len(res))
    else:
        rule = rule_rank_score(res)
    pick = dual_list(res["risk_score"].to_numpy(), rule, top_k, rule_quota)
    k_ai = int(round(min(top_k, len(res)) * (1 - rule_quota)))
    return res.index[pick[:k_ai]], res.index[pick[k_ai:]]


def cycle_fallback(res: pd.DataFrame, top_k: int, min_cycles: int) -> pd.DataFrame:
    """規則保底名單：AI 前 top_k 名以外、參與時間一致資金循環達 min_cycles 次的帳戶，另列專案查核。

    循環交易帳戶多為公司戶、很少被通報警示，監督式模型幾乎沒有樣本可學；R4 抓得到它們但單次循環誤報多，
    提高循環次數門檻後另列名單，補上 AI 的盲點（僅用規則結果，不含標籤）。
    """
    return res[(res["risk_rank"] > top_k) & (res["R4_cycles"] >= min_cycles)].sort_values("R4_cycles", ascending=False)


def cv_groups(accounts: pd.DataFrame, index: pd.Index) -> pd.Series | None:
    """集團層級切分用的群組：詐騙帳戶用集團代號，其他帳戶各自一組。"""
    if "fraud_group" not in accounts.columns:
        return None
    fg = accounts.set_index(accounts["account_id"].astype(str))["fraud_group"].reindex(index).fillna("").astype(str)
    if (fg == "").all():
        return None
    return pd.Series(np.where(fg != "", fg, index.to_numpy()), index=index)


def dump_json(obj, path: Path):
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
