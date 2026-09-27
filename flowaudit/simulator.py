"""台灣情境交易仿真器：產生「看起來像台灣銀行」的帳戶與交易資料。

為什麼需要它
------------
IBM AMLSim 的合成資料太單純（模型幾乎滿分），也沒有新台幣金額、交易時間、交易管道、
登入裝置、開戶日期等台灣銀行實務上最重要的欄位。本仿真器依公開資料設計參數，
產生包含「正常但看起來可疑」帳戶的資料，讓評估結果更接近真實。

仿真設計依據（詳見 docs/仿真設計與參數依據.md）
- 詐騙金流：被害人 → 第一層人頭帳戶 → 第二層人頭帳戶 → 車手 ATM 提領或轉入虛擬貨幣交易所，
  數小時內層層移轉（報導者 2023；聯合報 2026 屏東地檢署案）。
- 帳戶來源：新開戶、久未往來的休眠帳戶被收購、原本正常使用的帳戶被出售。
- 交易限額：ATM 跨行提款單次常見上限 2 萬元；非約定轉帳常見每筆 5 萬、每日 10 萬
  （各銀行不同，僅作為仿真參數）。
- 詐騙類型：假投資（金額大、被害人較晚發現）、網購詐騙（案件數最多、金額小）、
  假客服解除分期（夜間 ATM 轉帳、略低於限額）、循環交易（公司帳戶間資金空轉）。
- 警示時點：被害人報案後帳戶被通報警示並凍結；集團隨即換用下一個人頭帳戶。

困難的正常樣本（hard negatives）
- 房東：月初多名房客集中匯入租金（像集中匯入）。
- 團購主：短期內大量會員匯款，隨即整筆付給供應商（像集中匯入＋快進快出）。
- 公司發薪：同日匯給數十名員工（像分散匯出）。
- 休眠帳戶正常重新啟用：收到保險金或遺產後轉給家人（像休眠帳戶突然大額進出）。
- 家人共用手機、記帳士代管多家公司網銀（像共用裝置）。
- 一般民眾購買虛擬貨幣；被害人本身的大額匯款。

注意：本資料為合成資料，詐騙帳戶比例（約 1%）刻意高於真實情況，以利模型訓練與評估。
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
OUT_DIR = ROOT / "data" / "raw" / "tw_sim"

CASH = "CASH"  # 現金存提的對手方（不是帳戶）

# 一般民眾交易時段分布（0~23 時），深夜交易少
_HOUR_W = np.array([0.3, 0.2, 0.1, 0.1, 0.1, 0.2, 0.6, 1.5, 2.5, 3, 3, 3.2, 3.8, 3.2, 3, 3, 3, 3.2, 3.5, 4, 4, 3.5, 2.5, 1.2])
_HOUR_P = _HOUR_W / _HOUR_W.sum()

DEFAULT_CFG = {
    "seed": 20260101,
    "start": "2026-01-01",
    "months": 6,
    "n_office": 7200,
    "n_student": 2000,
    "n_retiree": 1800,
    "n_self": 1000,
    "n_landlord": 250,
    "n_groupbuy": 120,
    "n_merchant": 600,
    "n_employer": 100,
    "n_supplier": 50,
    "n_utility": 6,
    "n_exchange": 3,
    "n_reactivated": 80,
    "n_second": 2200,
    "n_collector": 150,
    "n_family_mgr": 120,
    "n_seller": 350,
    "n_rosca": 100,
    "fraud_groups": {"假投資": 7, "網購詐騙": 6, "解除分期": 4, "循環交易": 3},
    "atm_limit": 20000,
    "atm_daily_limit": 150000,
}


class TaiwanBankSimulator:
    def __init__(self, cfg: dict | None = None):
        self.cfg = {**DEFAULT_CFG, **(cfg or {})}
        c = self.cfg
        self.rng = np.random.default_rng(c["seed"])
        self.start = pd.Timestamp(c["start"])
        self.end = self.start + pd.DateOffset(months=c["months"])
        self.days = (self.end - self.start).days
        self.month_starts = [(self.start + pd.DateOffset(months=m) - self.start).days for m in range(c["months"] + 1)]
        self.acc: list[dict] = []
        self.idx: dict[str, int] = {}
        self.tx: list[tuple] = []
        self.inactive_until: dict[str, float] = {}
        self.inactive_after: dict[str, float] = {}
        self.night_shift: set[str] = set()
        self.mule_ctrl: dict[str, bool] = {}
        self.device: dict[str, str] = {}
        self.n_dev = 0
        self.groups: list[dict] = []

    # ------------------------------------------------------------------ 基本工具
    def u(self, a, b):
        return float(self.rng.uniform(a, b))

    def ri(self, a, b):
        return int(self.rng.integers(a, b + 1))

    def pick(self, seq, n=None, p=None):
        if n is None:
            return seq[int(self.rng.choice(len(seq), p=p))]
        idx = self.rng.choice(len(seq), size=min(n, len(seq)), replace=False, p=p)
        return [seq[i] for i in idx]

    def hour(self):
        return float(self.rng.choice(24, p=_HOUR_P) + self.rng.random())

    def t_on(self, day, hour=None):
        return day * 24 + (self.hour() if hour is None else hour)

    def new_device(self):
        self.n_dev += 1
        return f"D{self.n_dev:06d}"

    def new_account(self, ctype, role, open_days_before=None, dormant=0, own_device=True, **kw):
        aid = f"A{len(self.acc) + 1:06d}"
        if open_days_before is None:
            open_days_before = self.ri(365, 365 * 20)
        self.idx[aid] = len(self.acc)
        self.acc.append(dict(
            account_id=aid, customer_type=ctype, role=role,
            open_date=(self.start - pd.Timedelta(days=int(open_days_before))).normalize(),
            dormant_days_before=int(dormant), label=0, fraud_group="", scheme="", mule_layer=0,
            alert_date=pd.NaT, **kw,
        ))
        if own_device:
            self.device[aid] = self.new_device()
        return aid

    def add(self, t, src, dst, amount, channel, device="", fraud=0, pattern=""):
        amount = int(round(amount))
        if amount <= 0 or t < 0 or t >= self.days * 24 or src == dst:
            return
        iu = self.inactive_until
        if (src in iu and t < iu[src]) or (dst in iu and t < iu[dst]):
            return
        ia = self.inactive_after
        if (src in ia and t > ia[src]) or (dst in ia and t > ia[dst]):
            return
        if src in self.night_shift and channel in ("行動支付", "網路銀行", "行動銀行", "ATM提款") and self.rng.random() < 0.5:
            t = (t // 24) * 24 + self.u(0, 6)
        if channel in ("網路銀行", "行動銀行", "行動支付") and not device:
            device = self.device.get(src, "")
        self.tx.append((t, src, dst, amount, channel, device, fraud, pattern))

    def months(self):
        for m in range(self.cfg["months"]):
            yield m, self.month_starts[m], self.month_starts[m + 1]

    # ------------------------------------------------------------------ 正常帳戶
    def build_population(self):
        c = self.cfg
        mk = self.new_account
        self.employers = [mk("公司", "雇主") for _ in range(c["n_employer"])]
        self.suppliers = [mk("公司", "供應商") for _ in range(c["n_supplier"])]
        self.utilities = [mk("公司", "公用事業") for _ in range(c["n_utility"])]
        self.pension = mk("公司", "年金機構")
        self.epay = mk("公司", "電子支付機構")
        self.insurer = mk("公司", "保險公司")
        self.exchanges = [mk("公司", "虛擬貨幣交易所") for _ in range(c["n_exchange"])]
        self.merchants = [mk("商家", "商家") for _ in range(c["n_merchant"])]
        self.payday = {e: int(self.rng.choice([5, 10])) - 1 for e in self.employers}

        # 記帳士代管多家公司網銀：同一裝置被 3~6 家公司使用（正常的共用裝置）
        comps = self.employers + self.suppliers
        self.rng.shuffle(comps)
        i = 0
        for _ in range(20):
            k = self.ri(3, 6)
            dev = self.new_device()
            for a in comps[i:i + k]:
                self.device[a] = dev
            i += k

        self.office = [mk("個人", "上班族") for _ in range(c["n_office"])]
        self.student = [mk("個人", "學生") for _ in range(c["n_student"])]
        self.retiree = [mk("個人", "退休人士") for _ in range(c["n_retiree"])]
        self.selfemp = [mk("個人", "自營業者") for _ in range(c["n_self"])]
        self.landlords = [mk("個人", "房東") for _ in range(c["n_landlord"])]
        self.hosts = [mk("個人", "團購主") for _ in range(c["n_groupbuy"])]
        self.persons = self.office + self.student + self.retiree + self.selfemp + self.landlords + self.hosts

        # 新進員工：期間內才開戶（正常的新帳戶）
        self.new_start = {}
        for a in self.pick(self.office, int(0.05 * len(self.office))):
            d = self.ri(0, self.days - 40)
            self.acc[self.idx[a]]["open_date"] = (self.start + pd.Timedelta(days=d)).normalize()
            self.new_start[a] = d

        # 家人共用手機（2 個帳戶共用一個裝置）
        for a, b in zip(self.pick(self.persons, 900), self.pick(self.persons, 900)):
            if a != b:
                self.device[b] = self.device[a]

        # 休眠後正常重新啟用：收到保險金或遺產
        self.reactivated = {}
        for a in self.pick(self.retiree + self.office, c["n_reactivated"]):
            self.acc[self.idx[a]]["dormant_days_before"] = self.ri(200, 2000)
            self.reactivated[a] = self.ri(10, self.days - 20)

        self.persons_emp = self.persons + self.employers
        self.emp_sup = self.employers + self.suppliers
        self.sup_util = self.suppliers + self.utilities
        self.office_retiree = self.office + self.retiree
        self.inactive_until = {a: d * 24 for a, d in self.new_start.items()}
        self.inactive_until.update({a: d * 24 for a, d in self.reactivated.items()})
        self.family = {p: self.pick(self.persons, self.ri(1, 3)) for p in self.persons}
        self.fav_merchants = {p: self.pick(self.merchants, 8) for p in self.persons}
        self.employer_of = {p: self.pick(self.employers) for p in self.office}
        self.salary = {p: float(np.clip(self.rng.lognormal(np.log(45000), 0.35), 27000, 180000)) for p in self.office}
        renters = self.pick(self.office, int(0.4 * len(self.office))) + self.pick(self.student, int(0.3 * len(self.student)))
        self.rent = {p: (self.pick(self.landlords), round(self.u(6000, 25000), -2), self.ri(0, 4)) for p in renters}
        self.parent = {s: self.pick(self.office_retiree) for s in self.student}
        self.community = {h: self.pick(self.persons, self.ri(30, 120)) for h in self.hosts}
        self.crypto_buyers = set(self.pick(self.persons, int(0.06 * len(self.persons))))

        # 夜班工作者：深夜交易多（正常）
        self.night_shift = set(self.pick(self.persons, int(0.08 * len(self.persons))))
        # 期間內結清或不再使用的帳戶（使用期間短，正常）
        for p in self.pick(self.persons, int(0.05 * len(self.persons))):
            if p not in self.new_start and p not in self.reactivated:
                self.inactive_after[p] = self.ri(20, self.days - 10) * 24
        # 學生、退休人士、自營業者也有期間內新開戶
        for p in self.pick(self.student + self.retiree + self.selfemp, 250):
            if p in self.reactivated or p in self.inactive_after or p in self.new_start:
                continue
            d = self.ri(0, self.days - 30)
            self.acc[self.idx[p]]["open_date"] = (self.start + pd.Timedelta(days=d)).normalize()
            self.new_start[p] = d
            self.inactive_until[p] = d * 24
        # 第二帳戶（儲蓄、特定用途、極少使用）：交易稀疏、使用期間短，也常是久未往來的帳戶
        self.second = []
        for _ in range(c["n_second"]):
            owner = self.pick(self.persons)
            dormant = self.ri(180, 1500) if self.rng.random() < 0.4 else self.ri(0, 60)
            a = self.new_account("個人", "第二帳戶", dormant=dormant, own_device=False)
            self.device[a] = self.device[owner]
            kind = str(self.rng.choice(["儲蓄", "特定用途", "極少使用"], p=[0.55, 0.3, 0.15]))
            self.second.append((a, owner, kind))
        # 薪水入帳後立刻轉到自己的另一個帳戶（正常的快進快出）
        self.salary_fwd = {}
        for p in self.pick(self.office, int(0.25 * len(self.office))):
            a = self.new_account("個人", "第二帳戶", own_device=False)
            self.device[a] = self.device[p]
            self.salary_fwd[p] = a
        # 代收代付（班費、社團活動費）：短期內多人匯入後整筆付給廠商（正常）；一半用臨時開的帳戶收款
        self.collectors = []
        for p in self.pick(self.student + self.office, c["n_collector"]):
            if self.rng.random() < 0.5:
                d = self.ri(0, self.days - 20)
                a = self.new_account("個人", "代收帳戶", open_days_before=self.ri(0, 30) - d, own_device=False)
                self.device[a] = self.device[p]
                self.inactive_until[a] = max(d - 1, 0) * 24
                self.new_start[a] = d
                self.family[a] = self.family[p]
                self.collectors.append(a)
            else:
                self.collectors.append(p)
        # 個人賣家（網拍、二手、代購）：一段期間內很多不同買家小額匯款，再提領或轉出；常是新開或久未使用的帳戶
        self.sellers = []
        for _ in range(c["n_seller"]):
            kind = self.rng.choice(["新開戶", "休眠帳戶", "既有帳戶"], p=[0.5, 0.2, 0.3])
            d = self.ri(0, self.days - 25)
            if kind == "新開戶":
                a = self.new_account("個人", "個人賣家", open_days_before=self.ri(1, 40) - d)
            elif kind == "休眠帳戶":
                a = self.new_account("個人", "個人賣家", dormant=self.ri(180, 1200))
            else:
                a = self.new_account("個人", "個人賣家", dormant=self.ri(0, 30))
                d = 0
            if kind != "既有帳戶":
                self.inactive_until[a] = max(d - 1, 0) * 24
                self.new_start[a] = d
            self.family[a] = self.pick(self.persons, self.ri(1, 2))
            self.sellers.append((a, d, self.ri(20, 120)))
        # 標會的會頭：每月固定日期多名會員繳會款，隔天內把整筆會款交給得標會員（台灣常見的民間互助會）
        self.rosca = []
        for _ in range(c["n_rosca"]):
            head = self.pick(self.retiree + self.selfemp + self.office)
            members = self.pick(self.persons, self.ri(10, 20))
            self.rosca.append((head, members, round(self.u(5000, 20000), -3), self.ri(0, 20)))
        # 家長用同一支手機管理 3～4 個子女帳戶，每月付同一間補習班（正常的共用裝置＋同一受款人）
        self.family_mgr = []
        for _ in range(c["n_family_mgr"]):
            kids = self.pick(self.student, self.ri(3, 4))
            dev = self.new_device()
            for k in kids:
                self.device[k] = dev
            self.family_mgr.append((kids, self.pick(self.merchants)))

    def _active_from(self, p):
        return self.new_start.get(p, self.reactivated.get(p, 0))

    def normal_activity(self):
        add, u, ri = self.add, self.u, self.ri
        pois = self.rng.poisson
        for m, d0, d1 in self.months():
            dm = d1 - d0
            # 公司發薪（分散匯出）
            for p in self.office:
                if d0 + self.payday[self.employer_of[p]] >= self._active_from(p):
                    ts = self.t_on(d0 + self.payday[self.employer_of[p]], u(9, 11))
                    sal = self.salary[p] * u(0.98, 1.05)
                    add(ts, self.employer_of[p], p, sal, "薪資轉帳")
                    if p in self.salary_fwd:
                        add(ts + u(0.5, 20), p, self.salary_fwd[p], round(sal * u(0.6, 1.0), -2), self.pick(["網路銀行", "行動銀行"]))
            # 薪轉後的主要消費帳戶
            for p, a in self.salary_fwd.items():
                for _ in range(pois(3)):
                    add(self.t_on(ri(d0, d1 - 1)), a, self.pick(self.fav_merchants[p]), max(35, self.rng.lognormal(np.log(450), 0.8)), "行動支付")
                for _ in range(pois(1.5)):
                    add(self.t_on(ri(d0, d1 - 1)), a, CASH, 1000 * ri(1, 10), "ATM提款")
                if self.rng.random() < 0.5:
                    add(self.t_on(ri(d0, d1 - 1)), a, self.pick(self.family[p]), round(u(1000, 20000), -2), "行動銀行")
            # 第二帳戶：儲蓄
            for a, owner, kind in self.second:
                if kind == "儲蓄":
                    if self.rng.random() < 0.5:
                        add(self.t_on(ri(d0, d1 - 1)), owner, a, round(u(2000, 20000), -2), self.pick(["網路銀行", "行動銀行"]), device=self.device[owner])
                    if self.rng.random() < 0.1:
                        add(self.t_on(ri(d0, d1 - 1)), a, self.pick([owner, CASH]), round(u(5000, 50000), -3), self.pick(["網路銀行", "ATM提款"]))
            # 家長替子女付補習費
            for kids, tutor in self.family_mgr:
                for k in kids:
                    add(self.t_on(d0 + ri(0, 9)), k, tutor, round(u(3000, 8000), -2), "行動銀行")
            for p in self.retiree:
                if d0 + 4 >= self._active_from(p):
                    add(self.t_on(d0 + 4, u(9, 11)), self.pension, p, u(12000, 40000), "薪資轉帳")
            # 租金：月初集中匯給房東（集中匯入）
            for p, (ll, amt, dd) in self.rent.items():
                if d0 + dd >= self._active_from(p):
                    add(self.t_on(d0 + dd), p, ll, amt, self.pick(["網路銀行", "行動銀行", "ATM轉帳"]))
            for p in self.persons:
                a0 = self._active_from(p)
                if d1 <= a0:
                    continue
                lo = max(d0, a0)
                role = self.acc[self.idx[p]]["role"]
                # 行動支付消費
                for _ in range(pois(3 if role != "學生" else 2)):
                    add(self.t_on(ri(lo, d1 - 1)), p, self.pick(self.fav_merchants[p]),
                        max(35, self.rng.lognormal(np.log(450), 0.8)), "行動支付")
                # ATM 提款
                for _ in range(pois(1.5)):
                    add(self.t_on(ri(lo, d1 - 1)), p, CASH, 1000 * ri(1, 10 if role != "學生" else 3), "ATM提款")
                # 家人轉帳（包含互相轉來轉去的小循環）
                for _ in range(pois(0.8)):
                    add(self.t_on(ri(lo, d1 - 1)), p, self.pick(self.family[p]), round(u(1000, 20000), -2),
                        self.pick(["網路銀行", "行動銀行"]))
                # 水電瓦斯、電信自動扣繳
                if role != "學生":
                    add(self.t_on(ri(max(lo, d0 + 14), d1 - 1), u(1, 5)), p, self.pick(self.utilities), u(500, 3500), "自動扣繳")
                if p in self.crypto_buyers and self.rng.random() < 0.5:
                    add(self.t_on(ri(lo, d1 - 1)), p, self.pick(self.exchanges), round(u(3000, 60000), -3), "網路銀行")
                if p in self.crypto_buyers and self.rng.random() < 0.2:
                    add(self.t_on(ri(lo, d1 - 1)), self.pick(self.exchanges), p, round(u(5000, 80000), -2), "網路銀行")
                if role == "學生":
                    add(self.t_on(ri(d0, d0 + 2)), self.parent[p], p, round(u(3000, 12000), -2), "行動銀行")
                elif role == "退休人士":
                    for _ in range(pois(0.4)):
                        add(self.t_on(ri(lo, d1 - 1)), p, CASH, round(u(20000, 100000), -3), "臨櫃")
                elif role == "自營業者":
                    for _ in range(pois(3)):
                        add(self.t_on(ri(lo, d1 - 1)), self.pick(self.persons_emp), p, round(u(3000, 60000), -2), "網路銀行")
                    for _ in range(pois(2)):
                        add(self.t_on(ri(lo, d1 - 1)), p, self.pick(self.suppliers), round(u(5000, 80000), -2), "網路銀行")
                    for _ in range(pois(1)):
                        add(self.t_on(ri(lo, d1 - 1), u(9, 15)), CASH, p, round(u(10000, 100000), -3), "臨櫃")
                elif role == "房東":
                    add(self.t_on(ri(d0 + 6, d1 - 1)), p, self.pick(self.family[p]), round(u(20000, 80000), -3), "網路銀行")
            # 團購主：短期內大量會員匯款，隨即整筆付給供應商
            for h in self.hosts:
                for _ in range(ri(1, 3)):
                    s = ri(d0, d1 - 6)
                    members = self.pick(self.community[h], ri(15, min(60, len(self.community[h]))))
                    tot = 0
                    for mem in members:
                        amt = round(u(200, 3000), -1)
                        tot += amt
                        add(self.t_on(s + ri(0, 2)), mem, h, amt, self.pick(["行動支付", "網路銀行", "ATM轉帳"]))
                    add(self.t_on(s + ri(3, 4)), h, self.pick(self.suppliers), tot * u(0.85, 0.95), "網路銀行")
            # 商家：現金存入、付款給供應商、轉給負責人
            for mc in self.merchants:
                for w in range(d0, d1, 7):
                    add(self.t_on(min(w + ri(0, 6), d1 - 1), u(9, 16)), CASH, mc, round(u(5000, 60000), -2), self.pick(["臨櫃", "ATM存款"]))
                for _ in range(ri(2, 5)):
                    add(self.t_on(ri(d0, d1 - 1), u(9, 17)), mc, self.pick(self.suppliers), round(u(8000, 120000), -2), "網路銀行")
                add(self.t_on(ri(d1 - 5, d1 - 1)), mc, self.pick(self.persons), round(u(20000, 150000), -3), "網路銀行")
            # 公司之間的貨款往來
            for e in self.employers:
                for _ in range(ri(3, 10)):
                    add(self.t_on(ri(d0, d1 - 1), u(9, 17)), self.pick(self.emp_sup), e, round(u(100000, 3000000), -3), "網路銀行")
                for _ in range(ri(1, 4)):
                    add(self.t_on(ri(d0, d1 - 1), u(9, 17)), e, self.pick(self.sup_util), round(u(20000, 800000), -3), "網路銀行")
            for s_ in self.suppliers:
                for _ in range(ri(2, 6)):
                    add(self.t_on(ri(d0, d1 - 1), u(9, 17)), s_, self.pick(self.emp_sup), round(u(50000, 1500000), -3), "網路銀行")

        # 第二帳戶：特定用途（買車頭期款、學費等），短期內集中入帳後整筆付出；極少使用
        for a, owner, kind in self.second:
            if kind == "特定用途":
                d = self.ri(0, self.days - 10)
                tot = 0
                for _ in range(self.ri(1, 3)):
                    amt = round(self.u(30000, 600000), -3)
                    tot += amt
                    self.add(self.t_on(d + self.ri(0, 3)), self.pick([owner] + self.family.get(owner, [])), a, amt, "網路銀行")
                self.add(self.t_on(d + self.ri(3, 6), self.u(9, 16)), a, self.pick(self.emp_sup), round(tot * self.u(0.8, 1.0), -3), "網路銀行")
            elif kind == "極少使用":
                for _ in range(self.ri(1, 3)):
                    if self.rng.random() < 0.5:
                        self.add(self.t_on(self.ri(0, self.days - 1)), owner, a, round(self.u(500, 10000), -2), "行動銀行", device=self.device[owner])
                    else:
                        self.add(self.t_on(self.ri(0, self.days - 1)), a, CASH, 1000 * self.ri(1, 5), "ATM提款")
        # 代收代付：班費、活動費
        for col in self.collectors:
            for _ in range(self.ri(1, 2)):
                d = self.ri(self._active_from(col), self.days - 8)
                tot = 0
                for payer in self.pick(self.persons, self.ri(10, 40)):
                    amt = round(self.u(500, 3000), -1)
                    tot += amt
                    self.add(self.t_on(d + self.ri(0, 4)), payer, col, amt, self.pick(["行動銀行", "網路銀行", "ATM轉帳"]))
                self.add(self.t_on(d + self.ri(4, 7)), col, self.pick(self.merchants + self.suppliers), tot * self.u(0.9, 1.0), "網路銀行")

        # 個人賣家
        for a, d, dur in self.sellers:
            end = min(self.days - 1, d + dur)
            buyers = self.pick(self.persons, self.ri(15, 80))
            acc_amt, last_cash = 0.0, d
            for b in buyers:
                t = self.t_on(self.ri(d, end))
                amt = round(self.u(300, 8000), -1)
                self.add(t, b, a, amt, self.pick(["網路銀行", "行動銀行", "ATM轉帳"]))
            for dd in range(d + self.ri(2, 6), end + 1, self.ri(3, 8)):
                if self.rng.random() < 0.6:
                    for _ in range(self.ri(1, 4)):
                        self.add(self.t_on(dd), a, CASH, 1000 * self.ri(3, 20), "ATM提款")
                else:
                    self.add(self.t_on(dd), a, self.pick(self.suppliers + self.family[a]), round(self.u(3000, 60000), -2), "網路銀行")
            for _ in range(self.ri(0, 6)):
                self.add(self.t_on(self.ri(d, end)), a, self.pick(self.merchants), max(35, self.rng.lognormal(np.log(400), 0.7)), "行動支付")
        # 標會
        for head, members, amt, day in self.rosca:
            for m0 in range(len(self.month_starts) - 1):
                d = self.month_starts[m0] + day
                pot = 0
                for mem in members:
                    self.add(self.t_on(d + self.ri(0, 1)), mem, head, amt, self.pick(["網路銀行", "行動銀行", "ATM轉帳"]))
                    pot += amt
                winner = self.pick(members)
                if self.rng.random() < 0.8:
                    self.add(self.t_on(d + self.ri(1, 2)), head, winner, pot * self.u(0.85, 1.0), "網路銀行")
                else:
                    self.cash_out(None, head, self.t_on(d + self.ri(1, 2), self.u(9, 15)), pot * self.u(0.85, 1.0), pattern="")

        # 綁定電子支付或新增約定帳戶時的 1 元驗證、朋友間轉 1 元測試（正常的極小額交易）
        for p in self.pick(self.persons, int(0.08 * len(self.persons))):
            d = self.ri(self._active_from(p), self.days - 2)
            for _ in range(self.ri(2, 4)):
                t = self.t_on(d, self.u(8, 22))
                if self.rng.random() < 0.6:
                    self.add(t, self.epay, p, self.ri(1, 5), "網路銀行")
                else:
                    self.add(t, p, self.pick(self.family[p]), self.ri(1, 10), "行動銀行")

        # 休眠帳戶正常重新啟用：一筆大額保險金或遺產，之後轉給家人、臨櫃提領
        for p, d in self.reactivated.items():
            big = round(self.u(200000, 2000000), -3)
            self.add(self.t_on(d, self.u(9, 15)), self.insurer, p, big, "網路銀行")
            for _ in range(self.ri(1, 3)):
                self.add(self.t_on(d + self.ri(0, 5)), p, self.pick(self.family[p]), round(big * self.u(0.1, 0.4), -3), "網路銀行")
            self.add(self.t_on(d + self.ri(1, 10), self.u(9, 15)), p, CASH, round(big * self.u(0.05, 0.2), -3), "臨櫃")

    # ------------------------------------------------------------------ 人頭帳戶
    def new_mule(self, g: dict, layer: int, act_day: int) -> str:
        kind = self.rng.choice(["新開戶", "休眠帳戶", "出售帳戶"], p=[0.35, 0.40, 0.25])
        role = {1: "第一層人頭帳戶", 2: "第二層人頭帳戶", 3: "循環交易帳戶"}[layer]
        ctype = "公司" if (layer == 3 and self.rng.random() < 0.6) else "個人"
        if kind == "新開戶":
            a = self.new_account(ctype, role, open_days_before=self.ri(3, 45) - act_day, own_device=True)
        elif kind == "休眠帳戶":
            a = self.new_account(ctype, role, dormant=self.ri(180, 1500), own_device=True)
        else:
            a = self.new_account(ctype, role, dormant=self.ri(0, 20), own_device=True)
            # 出售前是正常使用的帳戶
            for d in range(0, self.days, 7):
                if self.rng.random() < (0.6 if d < act_day else 0.3):
                    self.add(self.t_on(d + self.ri(0, 6)), a, self.pick(self.merchants), max(35, self.rng.lognormal(np.log(400), 0.7)), "行動支付")
                if self.rng.random() < 0.3:
                    self.add(self.t_on(d + self.ri(0, 6)), a, CASH, 1000 * self.ri(1, 5), "ATM提款")
            for d in range(0, act_day, 30):
                self.add(self.t_on(d + self.ri(0, 3)), self.pick(self.persons), a, round(self.u(3000, 15000), -2), "行動銀行")
        if kind != "出售帳戶":
            self.inactive_until[a] = max(act_day - 3, 0) * 24
        rec = self.acc[self.idx[a]]
        rec.update(label=1, fraud_group=g["group_id"], scheme=g["scheme"], mule_layer=layer, mule_source=kind)
        # 約六成的人頭帳戶由集團統一以少數裝置操作，其餘由車手或帳戶提供者自己的手機操作
        self.mule_ctrl[a] = bool(self.rng.random() < 0.6)
        # 約半數收購後先做幾筆小額測試交易（近似測試行為）
        others = g["mules"]
        for _ in range(self.ri(2, 4) if self.rng.random() < 0.5 else 0):
            t = self.t_on(max(act_day - self.ri(1, 3), 0))
            cp = self.pick(others) if others else self.pick(self.persons)
            if self.rng.random() < 0.5:
                self.add(t, a, cp, self.ri(1, 30), "網路銀行", device=self.mule_dev(g, a), fraud=1, pattern="測試交易")
            else:
                self.add(t, cp, a, self.ri(1, 30), "網路銀行", device=self.mule_dev(g, cp), fraud=1, pattern="測試交易")
        g["mules"].append(a)
        return a

    def mule_dev(self, g, a):
        if self.mule_ctrl.get(a, False):
            return self.pick(g["devices"])
        return self.device.get(a, "")

    def cash_out(self, g, mule, t, amount, pattern="車手提領"):
        """車手在 ATM 分多筆提領（單筆上限、每日上限）。"""
        lim, daily = self.cfg["atm_limit"], self.cfg["atm_daily_limit"]
        left = int(amount // 1000 * 1000)
        day_used, cur_day = 0, int(t // 24)
        while left >= 1000:
            if int(t // 24) != cur_day:
                cur_day, day_used = int(t // 24), 0
            if day_used + lim > daily:
                t = (cur_day + 1) * 24 + self.u(8, 12)
                continue
            amt = min(lim, left)
            self.add(t, mule, CASH, amt, "ATM提款", fraud=1 if g is not None else 0, pattern=pattern)
            left -= amt
            day_used += amt
            t += self.u(0.02, 0.15)

    def layer2_for(self, g, t):
        day = int(t // 24)
        alive = [m for m in g["l2"] if self.l2_alert.get(m, 1e9) > day]
        if not alive or (len(alive) < 2 and self.rng.random() < 0.3):
            m = self.new_mule(g, 2, day)
            g["l2"].append(m)
            # 第二層通常較晚被追查到；部分在觀察期間內未被通報
            self.l2_alert[m] = day + self.ri(20, 70) if self.rng.random() < 0.6 else 10**9
            alive.append(m)
        return self.pick(alive)

    def forward_l2(self, g, m2, t, amount):
        if self.rng.random() < 0.6:
            self.cash_out(g, m2, t + self.u(0.5, 8), amount)
        else:
            self.add(t + self.u(0.5, 12), m2, self.pick(self.exchanges), amount * self.u(0.97, 1.0), "網路銀行",
                     device=self.mule_dev(g, m2), fraud=1, pattern="虛擬貨幣出金")

    def fraud_scheme(self, g: dict):
        sc = g["scheme"]
        rng, u, ri = self.rng, self.u, self.ri
        s0 = ri(0, min(100, max(1, self.days - 40)))
        s1 = min(self.days - 1, s0 + ri(50, 110))
        # 被害人與付款排程
        if sc == "假投資":
            pool, w = self.retiree + self.office, np.r_[np.full(len(self.retiree), 3.0), np.ones(len(self.office))]
            victims = self.pick(pool, ri(25, 55), p=w / w.sum())
        elif sc == "網購詐騙":
            pool = self.student + self.office
            victims = self.pick(pool, ri(50, 110))
        else:  # 解除分期
            victims = self.pick(self.office_retiree + self.student, ri(20, 45))
        pays = []  # (t, victim, amount, channel, report_day)
        # 網購、解除分期以「一波一波」的方式集中詐騙，同一個人頭帳戶短時間內收到多名被害人匯款
        waves = sorted(ri(s0, max(s0, s1 - 5)) for _ in range(max(1, len(victims) // (8 if sc == "網購詐騙" else 6))))
        for v in victims:
            d = ri(s0, max(s0, s1 - 5)) if sc == "假投資" else self.pick(waves) + ri(0, 1)
            vp = []
            if sc == "假投資":
                n = min(7, 1 + rng.poisson(2))
                days_ = [d]
                for _ in range(n - 1):
                    days_.append(days_[-1] + ri(1, 7))
                report = days_[-1] + ri(3, 30)
                for dd in days_:
                    if rng.random() < 0.2:
                        vp.append((self.t_on(dd, u(9, 15)), round(u(100000, 800000), -4), "臨櫃"))
                    else:
                        vp.append((self.t_on(dd), float(rng.choice([10000, 20000, 30000, 50000, 50000])),
                                   self.pick(["網路銀行", "行動銀行"])))
            elif sc == "網購詐騙":
                report = d + ri(1, 10)
                for k in range(2 if rng.random() < 0.2 else 1):
                    vp.append((self.t_on(d + k), round(u(1000, 15000), -1), self.pick(["網路銀行", "行動銀行", "ATM轉帳"])))
            else:
                report = d + ri(1, 3)  # 被害人通常隔天以後才發現並報案
                for k in range(ri(1, 3)):
                    vp.append((d * 24 + u(18, 23.5) + k * 0.3, float(rng.choice([29985, 29989, 19987, 9999, 29999])), "ATM轉帳"))
            g["victim_report"][v] = report
            pays += [(t, v, amt, ch, report) for t, amt, ch in vp]
        pays.sort(key=lambda x: x[0])

        cur, cur_alert, cur_n, cap = None, None, 0, 0
        for t, v, amt, ch, report in pays:
            day = int(t // 24)
            if cur is None or day >= cur_alert or cur_n >= cap:
                if cur is not None:
                    g["alert"][cur] = cur_alert
                cur = self.new_mule(g, 1, day)
                g["l1"].append(cur)
                cur_alert, cur_n = 10**9, 0
                cap = ri(8, 25) if sc == "假投資" else ri(10, 30)
            self.add(t, v, cur, amt, ch, device=self.device.get(v, ""), fraud=1, pattern="被害人匯入")
            cur_alert = min(cur_alert, report)
            cur_n += 1
            # 第一層：數小時內轉出
            # 轉出速度不一：多數數小時內，部分隔 1～3 天，部分只轉出一部分
            r_ = rng.random()
            tf = t + (u(0.3, 6) if r_ < 0.5 else u(24, 72) if r_ < 0.8 else u(2, 12))
            keep = u(0.6, 0.9) if r_ >= 0.8 else u(0.97, 1.0)
            if sc == "假投資" or (sc != "解除分期" and rng.random() < 0.4):
                total = amt * keep
                k = ri(1, 3) if total > 60000 else 1
                parts = rng.dirichlet(np.ones(k)) * total
                for p_ in parts:
                    m2 = self.layer2_for(g, tf)
                    self.add(tf, cur, m2, p_, "網路銀行", device=self.mule_dev(g, cur), fraud=1, pattern="分層轉出")
                    self.forward_l2(g, m2, tf, p_)
                    tf += u(0.05, 0.5)
            else:
                self.cash_out(g, cur, tf, amt * keep)
        if cur is not None:
            g["alert"][cur] = cur_alert
        g["n_victims"] = len(victims)
        g["victim_amount"] = float(sum(p[2] for p in pays))

    def cycle_scheme(self, g: dict):
        u, ri = self.u, self.ri
        s0 = ri(0, min(60, max(1, self.days - 20)))
        members = [self.new_mule(g, 3, s0) for _ in range(ri(4, 7))]
        dev = self.pick(g["devices"])
        d = s0
        while d < self.days - 5:
            route = self.pick(members, ri(3, len(members)))
            amt = round(u(300000, 3000000), -4)
            t = self.t_on(d, u(9, 16))
            for a, b in zip(route, route[1:] + route[:1]):
                self.add(t, a, b, amt, "網路銀行", device=dev, fraud=1, pattern="循環交易")
                amt *= u(0.985, 1.0)
                t += u(2, 30)
            # 偶爾有外部收入與支出，讓帳戶看起來像正常營運
            for a in members:
                if self.rng.random() < 0.5:
                    self.add(self.t_on(d + ri(0, 5), u(9, 17)), self.pick(self.employers), a, round(u(50000, 500000), -3), "網路銀行")
                if self.rng.random() < 0.4:
                    self.add(self.t_on(d + ri(0, 5), u(9, 17)), a, self.pick(self.sup_util), round(u(10000, 200000), -3), "網路銀行", device=dev)
            d += ri(5, 15)
        for a in members:
            g["alert"][a] = self.ri(60, 200) if self.rng.random() < 0.2 else 10**9
        g["n_victims"], g["victim_amount"] = 0, 0.0

    def build_fraud(self):
        self.l2_alert = {}
        k = 0
        for scheme, n in self.cfg["fraud_groups"].items():
            for _ in range(n):
                k += 1
                g = dict(group_id=f"G{k:02d}", scheme=scheme, mules=[], l1=[], l2=[], alert={}, victim_report={},
                         devices=[self.new_device() for _ in range(self.ri(2, 3))])
                if scheme == "循環交易":
                    self.cycle_scheme(g)
                else:
                    self.fraud_scheme(g)
                for m in g["l2"]:
                    g["alert"][m] = self.l2_alert[m]
                self.groups.append(g)

    # ------------------------------------------------------------------ 輸出
    def finalize(self):
        alert_t = {}
        for g in self.groups:
            for m, d in g["alert"].items():
                if d < self.days:
                    self.acc[self.idx[m]]["alert_date"] = (self.start + pd.Timedelta(days=int(d))).normalize()
                    alert_t[m] = d * 24 + 9  # 通報警示後凍結
        tx = pd.DataFrame(self.tx, columns=["t", "src", "dst", "amount", "channel", "device_id", "is_fraud", "pattern"])
        # 警示後帳戶凍結：移除凍結後的交易
        at_src = tx["src"].map(alert_t).fillna(np.inf)
        at_dst = tx["dst"].map(alert_t).fillna(np.inf)
        tx = tx[(tx["t"] < at_src) & (tx["t"] < at_dst)]
        tx = tx.sort_values("t", kind="stable").reset_index(drop=True)
        tx.insert(0, "timestamp", (self.start + pd.to_timedelta(tx["t"], unit="h")).dt.floor("min"))
        tx = tx.drop(columns="t")
        tx.insert(0, "tx_id", np.arange(1, len(tx) + 1))
        acc = pd.DataFrame(self.acc)
        acc["mule_source"] = acc.get("mule_source", pd.Series(dtype=object)).fillna("")
        # 期間內從未交易的帳戶不列入
        used = set(tx["src"]) | set(tx["dst"])
        acc = acc[acc["account_id"].isin(used)].reset_index(drop=True)
        groups = pd.DataFrame([
            {"group_id": g["group_id"], "scheme": g["scheme"], "n_mules": len(g["mules"]), "n_layer1": len(g["l1"]),
             "n_layer2": len(g["l2"]), "n_victims": g["n_victims"], "victim_amount": g["victim_amount"]}
            for g in self.groups
        ])
        return tx, acc, groups

    def run(self):
        self.build_population()
        self.normal_activity()
        self.build_fraud()
        return self.finalize()


def generate(out_dir: Path = OUT_DIR, cfg: dict | None = None, verbose: bool = True):
    sim = TaiwanBankSimulator(cfg)
    tx, acc, groups = sim.run()
    out_dir.mkdir(parents=True, exist_ok=True)
    tx.to_csv(out_dir / "transactions.csv.gz", index=False, compression="gzip")
    acc.to_csv(out_dir / "accounts.csv", index=False)
    groups.to_csv(out_dir / "fraud_groups.csv", index=False)
    if verbose:
        print(f"交易 {len(tx):,} 筆，帳戶 {len(acc):,} 個，人頭／可疑帳戶 {int(acc['label'].sum()):,} 個"
              f"（{acc['label'].mean():.2%}），詐騙交易 {int(tx['is_fraud'].sum()):,} 筆")
        print(groups.to_string(index=False))
    return tx, acc, groups


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description="產生台灣情境仿真交易資料")
    ap.add_argument("--seed", type=int, default=DEFAULT_CFG["seed"])
    args = ap.parse_args()
    generate(cfg={"seed": args.seed})
