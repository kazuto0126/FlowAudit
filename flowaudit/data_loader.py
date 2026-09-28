"""資料載入：把不同來源的交易資料轉成統一格式。

統一格式
--------
transactions DataFrame 欄位：
    tx_id   交易序號 (int)
    src     匯出帳戶 (str)
    dst     匯入帳戶 (str)
    amount  金額 (float)
    step    交易日序（第幾天，int，從 0 起算）
    date    日曆日期 (datetime64)
    extra   其他來源欄位（如幣別、交易方式）會原樣保留

accounts DataFrame 欄位：
    account_id  帳戶 (str)
    label       是否為已知人頭/洗錢帳戶 (0/1)，未知時為 NaN
                ※ 標籤只用於「模型訓練與成效評估」，規則引擎與圖分析完全不使用標籤。

支援的資料來源
--------------
1. IBM AMLSim 公開樣本（Apache-2.0），已隨專案附在 data/raw/。
2. IBM AMLworld（Kaggle: "IBM Transactions for Anti Money Laundering"），
   例如 HI-Small_Trans.csv，下載後用 load_amlworld() 讀取。
3. 任何含 src,dst,amount,(step|date) 欄位的 CSV，用 load_generic_csv() 讀取。
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = PROJECT_ROOT / "data" / "raw"

CASH = "CASH"  # 現金存提的對手方，不是帳戶
EXTERNAL_IDS = {CASH}

# 帳戶屬性中「銀行實際知道、可給模型使用」的欄位；其餘（角色、集團、警示日）只供驗證
ACCOUNT_ATTRS = ["customer_type", "open_date", "dormant_days_before"]
GROUND_TRUTH_COLS = ["role", "fraud_group", "scheme", "mule_layer", "mule_source", "alert_date"]

DATASETS = {
    "tw_sim": "台灣情境仿真資料（預設）",
    "amlsim_fanin_cycle": "IBM AMLSim fan-in + cycle 樣本",
    "amlsim_fanin": "IBM AMLSim fan-in 樣本",
    "amlsim_cycle": "IBM AMLSim cycle 樣本",
}

AMLSIM_DATASETS = {
    "amlsim_fanin_cycle": "20K_fanin200cycle200",
    "amlsim_fanin": "20K_fanin200",
    "amlsim_cycle": "20K_cycle200",
}


@dataclass
class Dataset:
    name: str
    transactions: pd.DataFrame
    accounts: pd.DataFrame

    @property
    def has_time(self) -> bool:
        d = self.transactions["date"]
        return bool((d != d.dt.normalize()).any())

    @property
    def has_alert_dates(self) -> bool:
        return "alert_date" in self.accounts.columns and self.accounts["alert_date"].notna().any()

    @property
    def has_labels(self) -> bool:
        return self.accounts["label"].notna().any()

    def summary(self) -> dict:
        acc = self.accounts
        return {
            "name": self.name,
            "n_accounts": int(len(acc)),
            "n_transactions": int(len(self.transactions)),
            "n_labeled_positive": int(acc["label"].fillna(0).sum()) if self.has_labels else None,
            "positive_rate": float(acc["label"].mean()) if self.has_labels else None,
            "date_start": str(self.transactions["date"].min().date()),
            "date_end": str(self.transactions["date"].max().date()),
            "total_amount": float(self.transactions["amount"].sum()),
        }


def _finalize(tx: pd.DataFrame, accounts: pd.DataFrame | None, base_date: str, name: str) -> Dataset:
    tx = tx.copy()
    tx["src"] = tx["src"].astype(str)
    tx["dst"] = tx["dst"].astype(str)
    tx["amount"] = pd.to_numeric(tx["amount"], errors="coerce")
    tx = tx.dropna(subset=["amount"])
    tx = tx[tx["amount"] > 0]
    if "date" not in tx.columns:
        tx["date"] = pd.Timestamp(base_date) + pd.to_timedelta(tx["step"].astype(int), unit="D")
    else:
        tx["date"] = pd.to_datetime(tx["date"])
    if "step" not in tx.columns:
        d0 = tx["date"].dt.normalize().min()
        tx["step"] = (tx["date"].dt.normalize() - d0).dt.days.astype(int)
    tx["step"] = tx["step"].astype(int)
    # 以「小時」為單位的時間軸（只有日期的資料視為當天 0 時）
    tx["t_hours"] = tx["step"] * 24.0 + tx["date"].dt.hour + tx["date"].dt.minute / 60.0
    tx = tx.sort_values(["t_hours", "src", "dst"], kind="stable").reset_index(drop=True)
    if "tx_id" in tx.columns:
        tx = tx.drop(columns="tx_id")
    tx.insert(0, "tx_id", np.arange(1, len(tx) + 1))

    all_ids = pd.Index(pd.unique(pd.concat([tx["src"], tx["dst"]], ignore_index=True))).difference(list(EXTERNAL_IDS))
    if accounts is None:
        accounts = pd.DataFrame({"account_id": all_ids, "label": np.nan})
    else:
        accounts = accounts.copy()
        accounts["account_id"] = accounts["account_id"].astype(str)
        missing = all_ids.difference(accounts["account_id"])
        if len(missing):
            accounts = pd.concat(
                [accounts, pd.DataFrame({"account_id": missing, "label": np.nan})], ignore_index=True
            )
    accounts = accounts[~accounts["account_id"].isin(EXTERNAL_IDS)]
    accounts = accounts.drop_duplicates("account_id").reset_index(drop=True)
    for c in ("open_date", "alert_date"):
        if c in accounts.columns:
            accounts[c] = pd.to_datetime(accounts[c])
    return Dataset(name=name, transactions=tx, accounts=accounts)


def load_twsim(folder: str | Path | None = None, name: str = "tw_sim") -> Dataset:
    """讀取台灣情境仿真資料（python -m flowaudit.simulator 產生）。"""
    folder = Path(folder) if folder else RAW_DIR / "tw_sim"
    if not (folder / "transactions.csv.gz").exists():
        from .simulator import generate
        generate(folder, verbose=False)
    tx = pd.read_csv(folder / "transactions.csv.gz", parse_dates=["timestamp"], dtype={"device_id": str, "pattern": str})
    tx = tx.rename(columns={"timestamp": "date"})
    tx["device_id"] = tx["device_id"].fillna("")
    if "pattern" in tx.columns:
        tx["pattern"] = tx["pattern"].fillna("")
    acc = pd.read_csv(folder / "accounts.csv", parse_dates=["open_date", "alert_date"])
    for c in ("fraud_group", "scheme", "mule_source"):
        if c in acc.columns:
            acc[c] = acc[c].fillna("")
    return _finalize(tx, acc, "2026-01-01", name=name)


def load_dataset(key: str, base_date: str = "2025-01-01") -> Dataset:
    if key == "tw_sim":
        return load_twsim()
    return load_amlsim(key, base_date)


def load_amlsim(key: str = "amlsim_fanin_cycle", base_date: str = "2025-01-01") -> Dataset:
    """讀取隨專案附帶的 IBM AMLSim 20K 帳戶樣本。"""
    folder = RAW_DIR / AMLSIM_DATASETS[key]
    nodes = pd.read_csv(folder / "nodes.csv")
    tx = pd.read_csv(folder / "transactions.csv")
    tx = tx.rename(columns={"sourceNodeId": "src", "targetNodeId": "dst", "value": "amount", "time": "step"})
    # 帳號加上前綴，避免與數字索引混淆
    tx["src"] = "A" + tx["src"].astype(str).str.zfill(5)
    tx["dst"] = "A" + tx["dst"].astype(str).str.zfill(5)
    accounts = pd.DataFrame(
        {
            "account_id": "A" + nodes["nodeid"].astype(str).str.zfill(5),
            "label": nodes["isFraud"].astype(int),
            "init_balance": nodes["init_balance"],
        }
    )
    return _finalize(tx, accounts, base_date, name=key)


def load_amlworld(path: str | Path, currency: str | None = "US Dollar", nrows: int | None = None) -> Dataset:
    """讀取 IBM AMLworld 格式（Kaggle 的 HI-Small_Trans.csv 等）。

    欄位：Timestamp, From Bank, Account, To Bank, Account.1, Amount Received,
          Receiving Currency, Amount Paid, Payment Currency, Payment Format, Is Laundering
    帳戶標籤：只要曾參與任一筆洗錢交易，即標為 1。
    currency：只保留指定支付幣別的交易，避免不同幣別金額混算；設為 None 則全部保留。
    """
    raw = pd.read_csv(path, nrows=nrows)
    raw.columns = [c.strip() for c in raw.columns]
    if currency is not None:
        raw = raw[raw["Payment Currency"] == currency]
    tx = pd.DataFrame(
        {
            "src": raw["From Bank"].astype(str) + "_" + raw["Account"].astype(str),
            "dst": raw["To Bank"].astype(str) + "_" + raw["Account.1"].astype(str),
            "amount": raw["Amount Paid"].astype(float),
            "date": pd.to_datetime(raw["Timestamp"], format="%Y/%m/%d %H:%M"),
            "currency": raw["Payment Currency"],
            "payment_format": raw["Payment Format"],
            "is_laundering_tx": raw["Is Laundering"].astype(int),
        }
    )
    # 自己轉給自己（同帳戶）的交易不構成資金移轉，排除
    tx = tx[tx["src"] != tx["dst"]]
    bad = tx.loc[tx["is_laundering_tx"] == 1, ["src", "dst"]].stack().unique()
    ids = pd.unique(pd.concat([tx["src"], tx["dst"]]))
    accounts = pd.DataFrame({"account_id": ids})
    accounts["label"] = accounts["account_id"].isin(set(bad)).astype(int)
    return _finalize(tx, accounts, base_date="1970-01-01", name=Path(path).stem)


def load_generic_csv(
    path_or_buffer, labels_path_or_buffer=None, base_date: str = "2025-01-01", name: str = "uploaded"
) -> Dataset:
    """讀取一般格式：必須有 src, dst, amount，以及 step 或 date 其中之一。

    標籤檔（選填）欄位：account_id, label
    """
    tx = pd.read_csv(path_or_buffer)
    tx.columns = [c.strip().lower() for c in tx.columns]
    alias = {"from": "src", "to": "dst", "source": "src", "target": "dst", "value": "amount", "time": "step"}
    tx = tx.rename(columns={k: v for k, v in alias.items() if k in tx.columns and v not in tx.columns})
    need = {"src", "dst", "amount"}
    if not need.issubset(tx.columns):
        raise ValueError(f"CSV 缺少必要欄位：{sorted(need - set(tx.columns))}")
    if "date" in tx.columns:
        tx["date"] = pd.to_datetime(tx["date"])
    elif "step" not in tx.columns:
        raise ValueError("CSV 必須包含 step（第幾天）或 date（日期）欄位")
    accounts = None
    if labels_path_or_buffer is not None:
        accounts = pd.read_csv(labels_path_or_buffer)
        accounts.columns = [c.strip().lower() for c in accounts.columns]
    return _finalize(tx, accounts, base_date, name=name)
