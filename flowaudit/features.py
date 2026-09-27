"""特徵工程：交易行為、資金流速、時間與管道、資金網路圖、帳戶屬性與裝置。

所有特徵都只由交易紀錄與「銀行實際知道」的帳戶屬性（客戶類型、開戶日、休眠天數）計算，
不使用任何標籤或仿真的真實角色，因此可直接套用在新資料上。
資料缺少某類欄位（例如沒有交易時間或裝置）時，該類特徵為 0，不影響其他特徵。
"""
from __future__ import annotations

import networkx as nx
import numpy as np
import pandas as pd

from .data_loader import CASH

FEATURE_NAMES_ZH = {
    # 交易行為
    "n_in": "匯入筆數",
    "n_out": "匯出筆數",
    "amt_in": "匯入總額",
    "amt_out": "匯出總額",
    "in_amt_mean": "單筆匯入平均金額",
    "in_amt_std": "匯入金額變異",
    "in_amt_max_ratio": "最大單筆匯入／平均匯入",
    "out_amt_mean": "單筆匯出平均金額",
    "out_amt_std": "匯出金額變異",
    "out_in_ratio": "匯出／匯入金額比",
    "net_flow_ratio": "淨流量占比",
    "n_src": "不重複匯入來源數",
    "n_dst": "不重複匯出對象數",
    "active_days": "有交易的天數",
    "span_days": "首末交易間隔天數",
    "tx_per_active_day": "每活動日交易筆數",
    "round_amount_share": "整千元交易比例",
    # 資金流速
    "lag_in_to_out": "流入後至下一筆轉出的平均小時數",
    "share_out_within_24h": "流入後 24 小時內即轉出的比例",
    "R1_event_share": "快進快出發生比例",
    "R1_event_amount_share": "快進快出金額占比",
    "R1_overall_ratio": "整體轉出／流入比",
    # 時間與管道
    "night_share": "深夜（0～6 時）交易比例",
    "cash_out_share": "現金提領占轉出比例",
    "cash_in_share": "現金存入占流入比例",
    "R6_n_withdrawals": "現金提領次數",
    "R6_max_count": "24 小時內最多提領次數",
    "R6_max_amount": "24 小時內最高提領金額",
    "atm_transfer_in_share": "ATM 轉帳匯入比例",
    "online_share": "網銀／行動交易比例",
    "salary_in_share": "薪資轉帳匯入比例",
    # 資金網路圖
    "R2_max_cp": "7 日內最多不同匯入來源",
    "R3_max_cp": "7 日內最多不同匯出對象",
    "R4_cycles": "參與資金循環次數",
    "pagerank": "資金網路重要性（PageRank）",
    "clustering": "交易對象彼此往來程度（聚集係數）",
    "reciprocity": "雙向往來對象比例",
    "nbr_in_degree_mean": "上游帳戶平均匯入來源數",
    "nbr_out_degree_mean": "下游帳戶平均匯出對象數",
    "two_hop_reach": "兩層內可觸及帳戶數",
    "core_number": "網路核心層級（k-core）",
    "nbr_rule_share": "往來對象命中規則比例",
    "nbr_pass_through_mean": "往來對象平均快進快出比例",
    "community_size": "所屬交易社群大小",
    "community_rule_share": "所屬社群命中規則比例",
    # 帳戶屬性與裝置
    "account_age": "首次交易時的開戶天數",
    "dormancy": "首次交易前未往來天數",
    "is_company": "公司戶",
    "is_merchant": "商家戶",
    "R5_first_in": "開始使用 14 日內匯入金額",
    "R5_first_ratio": "開始使用 14 日內轉出／匯入比",
    "R7_device_accounts": "登入裝置共用帳戶數",
    "R7_common_payee": "同裝置帳戶轉入同一收款人數",
    "n_devices": "使用裝置數",
    "R8_n_tiny": "極小額交易筆數",
    "R8_max_count": "72 小時內最多極小額交易",
    # 規則
    "rule_hits": "命中紅旗規則數",
}

FEATURE_GROUPS = {
    "交易行為": ["n_in", "n_out", "amt_in", "amt_out", "in_amt_mean", "in_amt_std", "in_amt_max_ratio", "out_amt_mean",
               "out_amt_std", "out_in_ratio", "net_flow_ratio", "n_src", "n_dst", "active_days", "span_days",
               "tx_per_active_day", "round_amount_share"],
    "資金流速": ["lag_in_to_out", "share_out_within_24h", "R1_event_share", "R1_event_amount_share", "R1_overall_ratio"],
    "時間與管道": ["night_share", "cash_out_share", "cash_in_share", "R6_n_withdrawals", "R6_max_count", "R6_max_amount",
                "atm_transfer_in_share", "online_share", "salary_in_share"],
    "資金網路圖": ["R2_max_cp", "R3_max_cp", "R4_cycles", "pagerank", "clustering", "reciprocity", "nbr_in_degree_mean",
               "nbr_out_degree_mean", "two_hop_reach", "core_number", "nbr_rule_share", "nbr_pass_through_mean",
               "community_size", "community_rule_share"],
    "帳戶屬性與裝置": ["account_age", "dormancy", "is_company", "is_merchant", "R5_first_in", "R5_first_ratio",
                  "R7_device_accounts", "R7_common_payee", "n_devices", "R8_n_tiny", "R8_max_count"],
    "規則命中數": ["rule_hits"],
}

RULE_FEATURES = ["R1_event_share", "R1_event_amount_share", "R1_overall_ratio", "R2_max_cp", "R3_max_cp", "R4_cycles",
                 "R5_first_in", "R5_first_ratio", "R6_n_withdrawals", "R6_max_count", "R6_max_amount",
                 "R7_device_accounts", "R7_common_payee", "R8_n_tiny", "R8_max_count", "rule_hits"]


def behavioral_features(tx: pd.DataFrame, idx: pd.Index) -> pd.DataFrame:
    """交易行為、資金流速、時間與管道特徵。現金存提計入流入流出，但不算交易對手。"""
    f = pd.DataFrame(index=idx)
    ins = tx[tx["dst"] != CASH]
    outs = tx[tx["src"] != CASH]
    g_in = ins.groupby("dst")["amount"]
    g_out = outs.groupby("src")["amount"]
    f["n_in"] = g_in.size()
    f["n_out"] = g_out.size()
    f["amt_in"] = g_in.sum()
    f["amt_out"] = g_out.sum()
    f["in_amt_mean"] = g_in.mean()
    f["in_amt_std"] = g_in.std()
    f["in_amt_max_ratio"] = g_in.max() / g_in.mean()
    f["out_amt_mean"] = g_out.mean()
    f["out_amt_std"] = g_out.std()
    a2a = tx[(tx["src"] != CASH) & (tx["dst"] != CASH)]
    f["n_src"] = a2a.groupby("dst")["src"].nunique()
    f["n_dst"] = a2a.groupby("src")["dst"].nunique()
    f = f.fillna(0)
    f["out_in_ratio"] = np.where(f["amt_in"] > 0, f["amt_out"] / f["amt_in"].clip(lower=1e-9), 0)
    tot = f["amt_in"] + f["amt_out"]
    f["net_flow_ratio"] = np.where(tot > 0, (f["amt_in"] - f["amt_out"]) / tot.clip(lower=1e-9), 0)

    both = pd.concat([
        tx[["src", "step", "t_hours", "amount"]].rename(columns={"src": "acct"}),
        tx[["dst", "step", "t_hours", "amount"]].rename(columns={"dst": "acct"}),
    ])
    both = both[both["acct"] != CASH]
    g = both.groupby("acct")
    f["active_days"] = g["step"].nunique().reindex(idx).fillna(0)
    f["span_days"] = (g["step"].max() - g["step"].min()).reindex(idx).fillna(0)
    f["tx_per_active_day"] = np.where(f["active_days"] > 0, g.size().reindex(idx).fillna(0) / f["active_days"].clip(lower=1), 0)
    both["round"] = (both["amount"] % 1000 == 0) & (both["amount"] >= 1000)
    f["round_amount_share"] = both.groupby("acct")["round"].mean().reindex(idx).fillna(0)
    hour = both["t_hours"] % 24
    both["night"] = hour < 6
    has_time = bool((hour != 0).any())
    f["night_share"] = both.groupby("acct")["night"].mean().reindex(idx).fillna(0) if has_time else 0.0

    # 流入 → 下一筆轉出（含提領）的時間差（小時）
    i_ = tx[tx["dst"] != CASH][["dst", "t_hours"]].rename(columns={"dst": "acct", "t_hours": "t_in"}).sort_values("t_in")
    o_ = tx[tx["src"] != CASH][["src", "t_hours"]].rename(columns={"src": "acct", "t_hours": "t_out"}).sort_values("t_out")
    m = pd.merge_asof(i_, o_, left_on="t_in", right_on="t_out", by="acct", direction="forward")
    m["lag"] = m["t_out"] - m["t_in"]
    lag = m.groupby("acct")["lag"].mean().reindex(idx)
    f["lag_in_to_out"] = lag.fillna(lag.max() if lag.notna().any() else 0)
    f["share_out_within_24h"] = m.assign(q=m["lag"] <= 24).groupby("acct")["q"].mean().reindex(idx).fillna(0)

    # 管道
    cash_out = tx[tx["dst"] == CASH].groupby("src")["amount"].sum().reindex(idx).fillna(0)
    cash_in = tx[tx["src"] == CASH].groupby("dst")["amount"].sum().reindex(idx).fillna(0)
    all_out = tx.groupby("src")["amount"].sum().reindex(idx).fillna(0)
    all_in = tx.groupby("dst")["amount"].sum().reindex(idx).fillna(0)
    f["cash_out_share"] = np.where(all_out > 0, cash_out / all_out.clip(lower=1e-9), 0)
    f["cash_in_share"] = np.where(all_in > 0, cash_in / all_in.clip(lower=1e-9), 0)
    if "channel" in tx.columns:
        ch = tx["channel"].astype(str)
        n_in_all = tx.groupby("dst").size().reindex(idx).fillna(0)
        f["atm_transfer_in_share"] = (tx[ch == "ATM轉帳"].groupby("dst").size().reindex(idx).fillna(0) / n_in_all.clip(lower=1))
        f["salary_in_share"] = (tx[ch == "薪資轉帳"].groupby("dst")["amount"].sum().reindex(idx).fillna(0) / all_in.clip(lower=1e-9))
        online = ch.isin(["網路銀行", "行動銀行", "行動支付"])
        n_out_all = tx.groupby("src").size().reindex(idx).fillna(0)
        f["online_share"] = tx[online].groupby("src").size().reindex(idx).fillna(0) / n_out_all.clip(lower=1)
    else:
        f["atm_transfer_in_share"] = 0.0
        f["salary_in_share"] = 0.0
        f["online_share"] = 0.0
    if "device_id" in tx.columns:
        d = tx[(tx["device_id"].astype(str) != "") & (tx["src"] != CASH)]
        f["n_devices"] = d.groupby("src")["device_id"].nunique().reindex(idx).fillna(0)
    else:
        f["n_devices"] = 0.0
    return f.fillna(0)


def attribute_features(accounts: pd.DataFrame, idx: pd.Index, rules: pd.DataFrame) -> pd.DataFrame:
    f = pd.DataFrame(index=idx)
    acc = accounts.set_index(accounts["account_id"].astype(str)).reindex(idx)
    f["account_age"] = rules["R5_account_age"].fillna(-1) if "R5_account_age" in rules else -1
    f["dormancy"] = rules["R5_dormancy"].fillna(-1) if "R5_dormancy" in rules else -1
    ct = acc["customer_type"].astype(str) if "customer_type" in acc.columns else pd.Series("", index=idx)
    f["is_company"] = (ct == "公司").astype(float)
    f["is_merchant"] = (ct == "商家").astype(float)
    return f


def graph_features(tx: pd.DataFrame, idx: pd.Index, rules: pd.DataFrame) -> tuple[pd.DataFrame, nx.DiGraph]:
    a2a = tx[(tx["src"] != CASH) & (tx["dst"] != CASH)]
    edges = a2a.groupby(["src", "dst"])["amount"].sum().reset_index()
    G = nx.DiGraph()
    G.add_nodes_from(idx)
    G.add_weighted_edges_from(edges.itertuples(index=False, name=None))
    U = G.to_undirected()

    f = pd.DataFrame(index=idx)
    f["pagerank"] = pd.Series(nx.pagerank(G, weight="weight"))
    f["clustering"] = pd.Series(nx.clustering(U))
    indeg = dict(G.in_degree())
    outdeg = dict(G.out_degree())
    recip, nin, nout, reach = {}, {}, {}, {}
    adj = {n: set(U.neighbors(n)) for n in U.nodes}
    for n in idx:
        preds = set(G.predecessors(n))
        succs = set(G.successors(n))
        nb = preds | succs
        recip[n] = len(preds & succs) / len(nb) if nb else 0.0
        nin[n] = np.mean([indeg[p] for p in preds]) if preds else 0.0
        nout[n] = np.mean([outdeg[s] for s in succs]) if succs else 0.0
        # 兩層內可觸及帳戶數（以鄰居的度數加總近似，避免大型商家造成運算爆量）
        reach[n] = len(nb) + sum(len(adj[x]) - 1 for x in nb)
    f["reciprocity"] = pd.Series(recip)
    f["nbr_in_degree_mean"] = pd.Series(nin)
    f["nbr_out_degree_mean"] = pd.Series(nout)
    f["two_hop_reach"] = pd.Series(reach)
    U.remove_edges_from(nx.selfloop_edges(U))
    f["core_number"] = pd.Series(nx.core_number(U))

    hit = (rules["rule_hits"] > 0).astype(float)
    pt = rules["R1_event_share"]
    hv, pv = hit.to_dict(), pt.to_dict()
    f["nbr_rule_share"] = pd.Series({n: np.mean([hv[x] for x in adj[n]]) if adj[n] else 0.0 for n in idx})
    f["nbr_pass_through_mean"] = pd.Series({n: np.mean([pv[x] for x in adj[n]]) if adj[n] else 0.0 for n in idx})

    comms = nx.community.louvain_communities(U, weight=None, seed=42, resolution=1.0)
    cid = {}
    for i, c in enumerate(comms):
        for n in c:
            cid[n] = i
    f["community_id"] = pd.Series(cid)
    f["community_size"] = f["community_id"].map(f["community_id"].value_counts())
    f["community_rule_share"] = f["community_id"].map(hit.groupby(f["community_id"]).mean())
    return f.fillna(0), G


def build_features(tx: pd.DataFrame, accounts: pd.DataFrame, rules: pd.DataFrame):
    idx = pd.Index(accounts["account_id"].astype(str), name="account_id")
    rules = rules.reindex(idx)
    b = behavioral_features(tx, idx)
    a = attribute_features(accounts, idx, rules)
    g, G = graph_features(tx, idx, rules)
    r = rules[RULE_FEATURES].astype(float)
    X = pd.concat([b, a, r, g.drop(columns=["community_id"])], axis=1).fillna(0)
    order = [c for grp in FEATURE_GROUPS.values() for c in grp]
    X = X[[c for c in order if c in X.columns]]
    meta = pd.DataFrame({"community_id": g["community_id"]}, index=idx)
    return X, meta, G
