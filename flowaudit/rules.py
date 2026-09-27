"""規則引擎：以稽核人員熟悉的「疑似洗錢／人頭帳戶表徵」逐一檢核每個帳戶。

設計原則
- 全查：每一個帳戶、每一筆交易都檢核，不做抽樣。
- 不使用標籤：規則只看交易行為，可直接用在沒有答案的新資料上。
- 有所本：每條規則對應法務部調查局「疑似洗錢或資恐交易態樣」銀行業代碼，
  或《存款帳戶及其疑似不法或顯屬異常交易管理辦法》第 4 條第二類情形。
- 可追溯：每一條命中的規則都附上證據文字，可直接寫入稽核工作底稿。
- 資料不足時自動降級：沒有時間、現金、裝置或開戶資料的欄位時，對應規則改用替代判斷或不觸發。
"""
from __future__ import annotations

from bisect import bisect_left, bisect_right
from collections import defaultdict

import numpy as np
import pandas as pd

from .data_loader import CASH

RULES = {
    "R1": "快進快出",
    "R2": "集中匯入",
    "R3": "分散匯出",
    "R4": "資金循環",
    "R5": "休眠／新戶突然活躍",
    "R6": "密集現金提領",
    "R7": "多帳戶共用裝置",
    "R8": "小額測試交易",
}

RULE_REFS = {
    "R1": "態樣 A17、A16、A15（款項存匯入後迅速移轉）",
    "R2": "態樣 A17（密集存入多筆款項）",
    "R3": "態樣 A18（經常於數個不同客戶帳戶間移轉資金）",
    "R4": "態樣 A18（資金於帳戶間循環移轉）",
    "R5": "態樣 A15、A16；管理辦法第 4 條（久未往來突有異常交易）",
    "R6": "態樣 A11、A12（一定期間內現金提款累計達特定金額）",
    "R7": "態樣 A1G（多個客戶網銀交易透過同一 IP，且密集轉入相同受款帳號）",
    "R8": "管理辦法第 4 條（多筆小額轉出入，近似測試行為）",
}

RULE_DESCRIPTIONS = {
    "R1": "資金流入後短時間內即大比例轉出或提領，帳戶僅作為過水使用，是人頭帳戶最典型的特徵。",
    "R2": "短期內有大量不同來源匯入同一帳戶，常見於詐騙集團收取多名被害人款項。",
    "R3": "短期內將資金分散匯出至大量不同帳戶，常見於洗錢的分層（layering）階段。",
    "R4": "資金依時間先後、金額相近地流經數個帳戶後回到原帳戶，用以製造虛假交易或混淆資金來源。",
    "R5": "新開戶或久未往來的帳戶，一開始使用就有大額資金進出且迅速轉走，常見於被收購的人頭帳戶。",
    "R6": "一天之內多次 ATM 提領且累計金額高，符合車手分批提領詐騙款項的行為。",
    "R7": "同一登入裝置操作多個帳戶，且這些帳戶都轉入同一受款帳號，顯示帳戶可能由同一人或集團控制。",
    "R8": "短期間內多筆極小額轉出入，常見於收購帳戶後測試帳戶是否可用。",
}


def _fmt_amt(x: float) -> str:
    return f"{x:,.0f}"


def _fmt_t(dates: pd.Series, t_hours: float, with_time: bool) -> str:
    d = dates.iloc[min(int(t_hours // 24), len(dates) - 1)]
    if with_time:
        h = int(t_hours % 24)
        m = int(round((t_hours % 1) * 60)) % 60
        return f"{d:%Y-%m-%d} {h:02d}:{m:02d}"
    return f"{d:%Y-%m-%d}"


class _Ctx:
    """規則共用的前處理結果。"""

    def __init__(self, tx: pd.DataFrame, accounts: pd.DataFrame):
        self.idx = pd.Index(accounts["account_id"].astype(str))
        self.pos = pd.Series(np.arange(len(self.idx)), index=self.idx)
        self.n = len(self.idx)
        d = tx["date"]
        self.with_time = bool((d != d.dt.normalize()).any())
        n_days = int(tx["step"].max()) + 1
        d0 = tx.loc[tx["step"] == tx["step"].min(), "date"].iloc[0].normalize() - pd.Timedelta(days=int(tx["step"].min()))
        self.dates = pd.Series(pd.date_range(d0, periods=n_days + 1, freq="D"))
        self.n_days = n_days
        self.src_pos = tx["src"].map(self.pos).to_numpy()
        self.dst_pos = tx["dst"].map(self.pos).to_numpy()
        self.is_cash_out = (tx["dst"] == CASH).to_numpy()
        self.is_cash_in = (tx["src"] == CASH).to_numpy()
        self.has_cash = bool(self.is_cash_out.any() or self.is_cash_in.any())
        self.t = tx["t_hours"].to_numpy()
        self.amt = tx["amount"].to_numpy(dtype=float)
        self.has_device = "device_id" in tx.columns and (tx["device_id"].astype(str) != "").any()
        self.has_attrs = {"open_date", "dormant_days_before"}.issubset(accounts.columns) and accounts["open_date"].notna().any()
        self.accounts = accounts.set_index(accounts["account_id"].astype(str))
        self.acct_tx = tx[~(self.is_cash_out | self.is_cash_in)]  # 帳戶對帳戶的交易


def _rolling_max(keys_acct: np.ndarray, t: np.ndarray, amt: np.ndarray, window: float, n: int):
    """每個帳戶在任一長度 window 的時間視窗內，最多幾筆、合計多少金額（向量化）。

    回傳 (max_count, max_amount, 發生最大筆數時的視窗起點時間)
    """
    max_cnt = np.zeros(n)
    max_amt = np.zeros(n)
    at_t = np.full(n, np.nan)
    if len(t) == 0:
        return max_cnt, max_amt, at_t
    BIG = 1e7
    order = np.lexsort((t, keys_acct))
    k, tt, aa = keys_acct[order], t[order], amt[order]
    key = k * BIG + tt
    cs = np.concatenate([[0.0], np.cumsum(aa)])
    j = np.searchsorted(key, key + window, side="right")
    i = np.arange(len(key))
    cnt = j - i
    tot = cs[j] - cs[i]
    df = pd.DataFrame({"k": k, "cnt": cnt, "tot": tot, "t": tt})
    g = df.sort_values(["k", "cnt", "tot"], ascending=[True, False, False]).drop_duplicates("k")
    max_cnt[g["k"].to_numpy()] = g["cnt"].to_numpy()
    at_t[g["k"].to_numpy()] = g["t"].to_numpy()
    max_amt_s = df.groupby("k")["tot"].max()
    max_amt[max_amt_s.index.to_numpy()] = max_amt_s.to_numpy()
    return max_cnt, max_amt, at_t


# ---------------------------------------------------------------------------
# R1 快進快出（以小時計）
# ---------------------------------------------------------------------------
def rule_pass_through(tx, c: _Ctx, cfg) -> pd.DataFrame:
    w = float(cfg["window_hours"])
    min_ratio = float(cfg["min_ratio"])
    in_mask = ~np.isnan(c.dst_pos)
    out_mask = ~np.isnan(c.src_pos)
    in_acct, in_t, in_amt = c.dst_pos[in_mask].astype(int), c.t[in_mask], c.amt[in_mask]
    out_acct, out_t, out_amt = c.src_pos[out_mask].astype(int), c.t[out_mask], c.amt[out_mask]
    # 忽略極小額的流入（避免測試交易、零錢造成大量事件）
    floor = np.percentile(in_amt, 25) if len(in_amt) else 0
    BIG = 1e7
    o = np.lexsort((out_t, out_acct))
    okey = out_acct[o] * BIG + out_t[o]
    cs = np.concatenate([[0.0], np.cumsum(out_amt[o])])
    ikey = in_acct * BIG + in_t
    lo = np.searchsorted(okey, ikey, side="left")
    hi = np.searchsorted(okey, ikey + w, side="right")
    fwd = cs[hi] - cs[lo]
    ratio = np.where(in_amt > 0, fwd / in_amt, 0)
    valid = in_amt >= floor
    event = valid & (ratio >= min_ratio)

    n_events = np.bincount(in_acct, weights=event, minlength=c.n)
    n_valid = np.bincount(in_acct, weights=valid, minlength=c.n)
    ev_amt = np.bincount(in_acct, weights=np.where(event, in_amt, 0), minlength=c.n)
    tot_in = np.bincount(in_acct, weights=in_amt, minlength=c.n)
    tot_out = np.bincount(out_acct, weights=out_amt, minlength=c.n)
    with np.errstate(divide="ignore", invalid="ignore"):
        share = np.where(n_valid > 0, n_events / np.maximum(n_valid, 1), 0.0)
        amt_share = np.where(tot_in > 0, ev_amt / np.maximum(tot_in, 1e-9), 0.0)
        overall = np.where(tot_in > 0, tot_out / np.maximum(tot_in, 1e-9), 0.0)
    hit = (n_events >= int(cfg["min_events"])) & (amt_share >= float(cfg["min_amount_share"]))

    evidence = np.full(c.n, "", dtype=object)
    ev_df = pd.DataFrame({"a": in_acct[event], "amt": in_amt[event], "t": in_t[event], "fwd": fwd[event]})
    best = ev_df.sort_values("amt", ascending=False).drop_duplicates("a").set_index("a")
    unit = f"{w:g} 小時" if c.with_time else f"{w / 24:g} 天"
    for i in np.flatnonzero(hit):
        b = best.loc[i]
        evidence[i] = (
            f"共 {int(n_events[i])} 筆流入（占流入金額 {amt_share[i]:.0%}）在 {unit}內被轉出或提領 ≥{min_ratio:.0%}；"
            f"例如 {_fmt_t(c.dates, b['t'], c.with_time)} 流入 {_fmt_amt(b['amt'])} 元，{unit}內轉出 {_fmt_amt(b['fwd'])} 元。"
            f"整體轉出／流入比 {overall[i]:.0%}。"
        )
    return pd.DataFrame(
        {"R1_hit": hit, "R1_events": n_events.astype(int), "R1_event_share": share, "R1_event_amount_share": amt_share,
         "R1_overall_ratio": overall, "R1_evidence": evidence},
        index=c.idx,
    )


# ---------------------------------------------------------------------------
# R2 / R3 集中匯入、分散匯出（視窗內不重複對手帳戶數）
# ---------------------------------------------------------------------------
def _max_distinct_in_window(tx: pd.DataFrame, key: str, other: str, window: int) -> pd.DataFrame:
    best = None
    for offset in (0, window // 2):
        b = (tx["step"] + offset) // window
        g = (
            pd.DataFrame({"acct": tx[key], "cp": tx[other], "bucket": b, "amt": tx["amount"]})
            .groupby(["acct", "bucket"])
            .agg(n_cp=("cp", "nunique"), amt=("amt", "sum"))
            .reset_index()
        )
        g["start_step"] = g["bucket"] * window - offset
        g = g.sort_values(["acct", "n_cp"], ascending=[True, False]).drop_duplicates("acct")
        best = g if best is None else pd.concat([best, g]).sort_values(["acct", "n_cp"], ascending=[True, False]).drop_duplicates("acct")
    return best.set_index("acct")


def rule_fan(tx, c: _Ctx, cfg, direction: str) -> pd.DataFrame:
    code = "R2" if direction == "in" else "R3"
    key, other = ("dst", "src") if direction == "in" else ("src", "dst")
    w = int(cfg["window_days"])
    m = _max_distinct_in_window(c.acct_tx, key, other, w).reindex(c.idx)
    n_cp = m["n_cp"].fillna(0).astype(int)
    # 依客群分別設定門檻（個人、商家、公司的正常往來對象數差很多，混在一起會讓個人戶門檻過高）
    seg = c.accounts.reindex(c.idx)["customer_type"].fillna("未分類").astype(str) if "customer_type" in c.accounts.columns \
        else pd.Series("全部", index=c.idx)
    thr = pd.Series(float(cfg["min_count"]), index=c.idx)
    for sname, members in n_cp.groupby(seg):
        active = members[members > 0]
        if len(active):
            thr[members.index] = max(int(cfg["min_count"]), int(np.ceil(np.percentile(active, cfg["percentile"]))))
    hit = n_cp >= thr
    verb = "由" if direction == "in" else "向"
    act = "匯入" if direction == "in" else "匯出"
    evidence = pd.Series("", index=c.idx, dtype=object)
    for a in hit[hit].index:
        s = int(max(m.at[a, "start_step"], 0))
        evidence[a] = (
            f"{c.dates.iloc[min(s, len(c.dates) - 1)]:%Y-%m-%d} 起 {w} 日內{verb} {n_cp[a]} 個不同帳戶{act}，"
            f"合計 {_fmt_amt(m.at[a, 'amt'])} 元；門檻為 {int(thr[a])} 個（{seg[a]}客群第 {cfg['percentile']} 百分位）。"
        )
    return pd.DataFrame(
        {f"{code}_hit": hit.values, f"{code}_max_cp": n_cp.values, f"{code}_threshold": thr.values, f"{code}_evidence": evidence.values},
        index=c.idx,
    )


# ---------------------------------------------------------------------------
# R4 資金循環（時間先後一致、金額相近的短循環）
# ---------------------------------------------------------------------------
def find_temporal_cycles(tx: pd.DataFrame, max_len: int = 5, max_span: float = 30, max_cycles: int = 20000,
                         amount_ratio: tuple[float, float] | None = None, min_amount: float = 0.0,
                         reach2_cap: int = 200_000):
    """找出「依時間先後」資金流回原帳戶的循環。

    從每一筆交易 u→v(t0) 出發，只沿著時間 ≥ 前一筆、且 ≤ t0 + max_span 天的交易往下走；
    若設定 amount_ratio=(lo, hi)，下一筆金額須介於前一筆的 lo～hi 倍（同一筆錢在流轉）。
    在 max_len 步內回到 u 即為一個時間一致的資金循環。回傳 list[(帳戶序列, 交易序號序列)]
    """
    t_col = "t_hours" if "t_hours" in tx.columns else "step"
    span = max_span * 24 if t_col == "t_hours" else max_span
    sub = tx[(tx["src"] != tx["dst"]) & (tx["src"] != CASH) & (tx["dst"] != CASH) & (tx["amount"] >= min_amount)]
    out = defaultdict(list)
    pred = defaultdict(set)       # 曾轉帳給該帳戶的帳戶
    succ = defaultdict(set)       # 該帳戶曾轉帳給的帳戶
    in_times = defaultdict(list)  # 該帳戶每筆流入的時間
    for u, v, s, a, i in zip(sub["src"].to_numpy(), sub["dst"].to_numpy(), sub[t_col].to_numpy(),
                             sub["amount"].to_numpy(dtype=float), sub["tx_id"].to_numpy()):
        out[u].append((float(s), v, float(a), int(i)))
        pred[v].add(u)
        succ[u].add(v)
        in_times[v].append(float(s))
    times = {}
    for u in out:
        out[u].sort()
        times[u] = [e[0] for e in out[u]]
    for v in in_times:
        in_times[v].sort()
    lo_r, hi_r = amount_ratio if amount_ratio else (0.0, np.inf)

    # 以下剪枝只略過「不可能回到起點」的搜尋分支，找到的循環與順序和逐一搜尋完全相同，大資料時快很多：
    # 1. 起點帳戶在時間視窗內沒有任何流入，就不可能回到起點；
    # 2. 下一個帳戶還剩 k 步可走時，必須能在 k 步內走回起點（k = 1、2、3；只看有沒有轉帳關係）。
    #    「2 步內能走回起點」的帳戶集合太大時不建立，只影響速度、不影響結果。
    empty = frozenset()
    found, seen = [], set()
    for u in list(out.keys()):
        if u not in pred:
            continue
        pu, itu = pred[u], in_times[u]
        reach2 = None  # 2 步內能走回 u 的帳戶（第一次用到時才建立）
        for s0, v0, a0, t0 in out[u]:
            limit = s0 + span
            if bisect_right(itu, limit) == bisect_left(itu, s0):
                continue
            if reach2 is None:
                size = len(pu) + sum(len(pred[p]) for p in pu if p in pred)
                reach2 = pu.union(*(pred[p] for p in pu if p in pred)) if size <= reach2_cap else False
            stack = [(v0, s0, a0, (u, v0), (t0,))]
            done = False
            while stack and not done:
                x, t, a, path, txs = stack.pop()
                if x not in out:
                    continue
                lo = bisect_left(times[x], t)
                hi = bisect_right(times[x], limit)
                steps_left = max_len - len(path)  # 下一個帳戶最多還能走幾步回到起點
                for s, y, amt, tid in out[x][lo:hi]:
                    if not (lo_r * a <= amt <= hi_r * a):
                        continue
                    if y == u:
                        key = frozenset(path)
                        if key not in seen:
                            seen.add(key)
                            found.append((list(path), list(txs) + [tid]))
                        done = True
                        break
                    if steps_left <= 0 or y in path:
                        continue
                    if steps_left == 1 and y not in pu:
                        continue
                    if steps_left == 2 and y not in pu and (y not in reach2 if reach2 else succ.get(y, empty).isdisjoint(pu)):
                        continue
                    if steps_left == 3 and reach2 and y not in pu and succ.get(y, empty).isdisjoint(reach2):
                        continue
                    stack.append((y, s, amt, path + (y,), txs + (tid,)))
            if len(found) >= max_cycles:
                return found
    return found


def rule_cycle(tx, c: _Ctx, cfg) -> tuple[pd.DataFrame, list]:
    ar = cfg.get("amount_ratio")
    cycles = find_temporal_cycles(tx, int(cfg["max_length"]), float(cfg["max_span_days"]),
                                  amount_ratio=tuple(ar) if ar else None, min_amount=float(cfg.get("min_amount", 0)))
    count = defaultdict(int)
    example = {}
    for nodes, txs in cycles:
        for n in nodes:
            count[n] += 1
            if n not in example or len(nodes) > len(example[n][0]):
                example[n] = (nodes, txs)
    cnt = pd.Series(count, dtype=float).reindex(c.idx).fillna(0).astype(int)
    hit = cnt >= int(cfg.get("min_cycles", 1))
    evidence = pd.Series("", index=c.idx, dtype=object)
    tx_by_id = tx.set_index("tx_id")
    for a in hit[hit].index:
        nodes, txs = example[a]
        k = nodes.index(a)
        order = nodes[k:] + nodes[:k] + [a]
        rows = tx_by_id.loc[txs]
        span = f"{rows['date'].min():%Y-%m-%d} ~ {rows['date'].max():%Y-%m-%d}"
        evidence[a] = (
            f"參與 {cnt[a]} 個時間一致的資金循環；例：{' → '.join(order)}"
            f"（{len(nodes)} 個帳戶，{span}，金額 {_fmt_amt(rows['amount'].min())}～{_fmt_amt(rows['amount'].max())} 元）。"
        )
    df = pd.DataFrame({"R4_hit": hit.values, "R4_cycles": cnt.values, "R4_evidence": evidence.values}, index=c.idx)
    return df, cycles


# ---------------------------------------------------------------------------
# R5 休眠／新戶突然活躍（沒有開戶資料時改用「活動驟增」）
# ---------------------------------------------------------------------------
def rule_dormant_new(tx, c: _Ctx, cfg) -> pd.DataFrame:
    both = pd.concat([
        pd.DataFrame({"a": c.src_pos, "t": c.t}), pd.DataFrame({"a": c.dst_pos, "t": c.t})
    ]).dropna()
    both["a"] = both["a"].astype(int)
    first_t = np.full(c.n, np.nan)
    ft = both.groupby("a")["t"].min()
    first_t[ft.index.to_numpy()] = ft.to_numpy()

    if not c.has_attrs:
        # 替代規則：活動驟增（單週交易筆數遠高於帳戶平均）
        wk = tx["step"] // 7
        n_weeks = int(wk.max()) + 1
        bw = pd.concat([pd.DataFrame({"acct": tx["src"], "wk": wk}), pd.DataFrame({"acct": tx["dst"], "wk": wk})])
        bw = bw[bw["acct"] != CASH]
        weekly = bw.groupby(["acct", "wk"]).size()
        mx = weekly.groupby(level=0).max().reindex(c.idx).fillna(0)
        mean = (weekly.groupby(level=0).sum() / n_weeks).reindex(c.idx).fillna(0)
        ratio = np.where(mean > 0, mx / np.maximum(mean, 1e-9), 0.0)
        hit = ((mx >= cfg["spike_min_weekly_tx"]) & (ratio >= cfg["spike_ratio"])).to_numpy()
        ev = np.full(c.n, "", dtype=object)
        for i in np.flatnonzero(hit):
            ev[i] = f"單週交易 {int(mx.iloc[i])} 筆，為該帳戶週平均 {mean.iloc[i]:.1f} 筆的 {ratio[i]:.1f} 倍（無開戶資料，改用活動驟增判斷）。"
        return pd.DataFrame({"R5_hit": hit, "R5_new_or_dormant": 0, "R5_first_in": 0.0, "R5_first_ratio": ratio,
                             "R5_account_age": np.nan, "R5_dormancy": np.nan, "R5_evidence": ev}, index=c.idx)

    acc = c.accounts.reindex(c.idx)
    first_day = np.floor(np.nan_to_num(first_t, nan=0) / 24)
    first_date = c.dates.iloc[0] + pd.to_timedelta(first_day, unit="D")
    age = (first_date - pd.to_datetime(acc["open_date"]).to_numpy()).days.to_numpy().astype(float)
    dormancy = acc["dormant_days_before"].fillna(0).to_numpy(dtype=float) + first_day
    new_or_dorm = (age <= cfg["new_account_days"]) | (dormancy >= cfg["dormant_days"])

    # 開始使用後 window 天內的流入與轉出
    w = cfg["window_days"] * 24
    ft_src = first_t[np.nan_to_num(c.src_pos, nan=0).astype(int)]
    ft_dst = first_t[np.nan_to_num(c.dst_pos, nan=0).astype(int)]
    in_m = ~np.isnan(c.dst_pos) & (c.t <= ft_dst + w)
    out_m = ~np.isnan(c.src_pos) & (c.t <= ft_src + w)
    fin = np.bincount(c.dst_pos[in_m].astype(int), weights=c.amt[in_m], minlength=c.n)
    fout = np.bincount(c.src_pos[out_m].astype(int), weights=c.amt[out_m], minlength=c.n)
    ratio = np.where(fin > 0, fout / np.maximum(fin, 1e-9), 0.0)
    hit = new_or_dorm & (fin >= cfg["min_inflow"]) & (ratio >= cfg["min_out_ratio"])
    ev = np.full(c.n, "", dtype=object)
    for i in np.flatnonzero(hit):
        why = f"開戶 {int(age[i])} 天" if age[i] <= cfg["new_account_days"] else f"久未往來 {int(dormancy[i])} 天"
        ev[i] = (f"{why}後開始使用，{cfg['window_days']} 日內匯入 {_fmt_amt(fin[i])} 元，"
                 f"同期轉出或提領 {_fmt_amt(fout[i])} 元（{ratio[i]:.0%}）。")
    return pd.DataFrame({"R5_hit": hit, "R5_new_or_dormant": new_or_dorm.astype(int), "R5_first_in": fin,
                         "R5_first_ratio": ratio, "R5_account_age": age, "R5_dormancy": dormancy, "R5_evidence": ev},
                        index=c.idx)


# ---------------------------------------------------------------------------
# R6 密集現金提領（車手）
# ---------------------------------------------------------------------------
def rule_cash(tx, c: _Ctx, cfg) -> pd.DataFrame:
    m = c.is_cash_out & ~np.isnan(c.src_pos)
    cnt, tot, at_t = _rolling_max(c.src_pos[m].astype(int), c.t[m], c.amt[m], float(cfg["window_hours"]), c.n)
    hit = (cnt >= cfg["min_count"]) & (tot >= cfg["min_amount"])
    ev = np.full(c.n, "", dtype=object)
    for i in np.flatnonzero(hit):
        ev[i] = (f"{_fmt_t(c.dates, at_t[i], c.with_time)} 起 {cfg['window_hours']} 小時內現金提領 {int(cnt[i])} 次，"
                 f"期間內單一視窗最高累計 {_fmt_amt(tot[i])} 元。")
    n_w = np.bincount(c.src_pos[m].astype(int), minlength=c.n)
    return pd.DataFrame({"R6_hit": hit, "R6_max_count": cnt, "R6_max_amount": tot, "R6_n_withdrawals": n_w,
                         "R6_evidence": ev}, index=c.idx)


# ---------------------------------------------------------------------------
# R7 多帳戶共用裝置，且轉入相同受款帳號（態樣 A1G）
# ---------------------------------------------------------------------------
def rule_device(tx, c: _Ctx, cfg) -> pd.DataFrame:
    empty = pd.DataFrame({"R7_hit": False, "R7_device_accounts": 0, "R7_common_payee": 0, "R7_evidence": ""}, index=c.idx)
    if not c.has_device:
        return empty
    t = tx[(tx["device_id"].astype(str) != "") & tx["src"].isin(c.idx)]
    dev_accts = t.groupby("device_id")["src"].nunique()
    max_dev = t.assign(n=t["device_id"].map(dev_accts)).groupby("src")["n"].max().reindex(c.idx).fillna(0)
    t2 = t[t["dst"] != CASH]
    pay = t2.groupby(["device_id", "dst"])["src"].nunique().rename("k").reset_index()
    pay = pay[pay["k"] >= cfg["min_accounts"]]
    flagged = t2.merge(pay, on=["device_id", "dst"])
    best = flagged.sort_values("k", ascending=False).drop_duplicates("src").set_index("src")
    common = best["k"].reindex(c.idx).fillna(0)
    hit = common >= cfg["min_accounts"]
    ev = pd.Series("", index=c.idx, dtype=object)
    for a in hit[hit].index:
        b = best.loc[a]
        ev[a] = (f"登入裝置 {b['device_id']} 共被 {int(dev_accts[b['device_id']])} 個帳戶使用，"
                 f"其中 {int(b['k'])} 個帳戶都轉入同一受款帳號 {b['dst']}。")
    return pd.DataFrame({"R7_hit": hit.to_numpy(), "R7_device_accounts": max_dev.to_numpy(), "R7_common_payee": common.to_numpy(),
                         "R7_evidence": ev.to_numpy()}, index=c.idx)


# ---------------------------------------------------------------------------
# R8 小額測試交易
# ---------------------------------------------------------------------------
def rule_tiny(tx, c: _Ctx, cfg) -> pd.DataFrame:
    tiny = (c.amt <= cfg["max_amount"]) & ~c.is_cash_out & ~c.is_cash_in
    a = np.concatenate([c.src_pos[tiny], c.dst_pos[tiny]])
    t = np.concatenate([c.t[tiny], c.t[tiny]])
    am = np.concatenate([c.amt[tiny], c.amt[tiny]])
    ok = ~np.isnan(a)
    cnt, _, at_t = _rolling_max(a[ok].astype(int), t[ok], am[ok], float(cfg["window_hours"]), c.n)
    n_tiny = np.bincount(a[ok].astype(int), minlength=c.n)
    hit = cnt >= cfg["min_count"]
    ev = np.full(c.n, "", dtype=object)
    for i in np.flatnonzero(hit):
        ev[i] = (f"{_fmt_t(c.dates, at_t[i], c.with_time)} 起 {cfg['window_hours']} 小時內有 {int(cnt[i])} 筆 "
                 f"{cfg['max_amount']} 元以下的極小額轉出入。")
    return pd.DataFrame({"R8_hit": hit, "R8_max_count": cnt, "R8_n_tiny": n_tiny, "R8_evidence": ev}, index=c.idx)


# ---------------------------------------------------------------------------
def run_rules(tx: pd.DataFrame, accounts: pd.DataFrame, cfg: dict) -> tuple[pd.DataFrame, list]:
    """執行全部規則，回傳 (每帳戶規則結果, 找到的資金循環清單)。"""
    c = _Ctx(tx, accounts)
    r1 = rule_pass_through(tx, c, cfg["pass_through"])
    r2 = rule_fan(tx, c, cfg["fan_in"], "in")
    r3 = rule_fan(tx, c, cfg["fan_out"], "out")
    r4, cycles = rule_cycle(tx, c, cfg["cycle"])
    r5 = rule_dormant_new(tx, c, cfg["dormant_new"])
    r6 = rule_cash(tx, c, cfg["cash"])
    r7 = rule_device(tx, c, cfg["device"])
    r8 = rule_tiny(tx, c, cfg["tiny"])
    out = pd.concat([r1, r2, r3, r4, r5, r6, r7, r8], axis=1)
    hits = out[[f"{r}_hit" for r in RULES]].astype(bool)
    out[[f"{r}_hit" for r in RULES]] = hits
    out["rule_hits"] = hits.sum(axis=1).astype(int)
    names = np.array(list(RULES.values()))
    hv = hits.to_numpy()
    out["rule_list"] = ["、".join(names[row]) for row in hv]
    out.index.name = "account_id"
    return out, cycles


def rule_availability(tx: pd.DataFrame, accounts: pd.DataFrame) -> dict:
    """各規則在這份資料上是否能完整運作（供儀表板說明）。"""
    c = _Ctx(tx, accounts)
    return {
        "R1": True, "R2": True, "R3": True, "R4": True,
        "R5": bool(c.has_attrs), "R6": bool(c.has_cash), "R7": bool(c.has_device), "R8": True,
    }
