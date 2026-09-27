"""基本測試：pytest -q"""
import io

import numpy as np
import pandas as pd
import pytest

from flowaudit import data_loader as dl
from flowaudit.pipeline import analyze, load_config, load_run
from flowaudit.report import (build_workpaper, collect_facts, generate_report, markdown_to_docx, render_template)
from flowaudit.rules import find_temporal_cycles, run_rules

CFG = load_config()


def _run_or_skip(key):
    from flowaudit.pipeline import OUT_DIR
    if not (OUT_DIR / key / "results.pkl").exists():
        pytest.skip(f"尚未執行 python -m flowaudit.pipeline --dataset {key}")
    return load_run(key)


def _tx(rows):
    df = pd.DataFrame(rows, columns=["src", "dst", "amount", "step"])
    return dl._finalize(df, None, "2025-01-01", "t")


def _txt(rows):
    """rows: (src, dst, amount, 'YYYY-MM-DD HH:MM', channel, device)"""
    df = pd.DataFrame(rows, columns=["src", "dst", "amount", "date", "channel", "device_id"])
    return dl._finalize(df, None, "2026-01-01", "t")


def _noise(n=60):
    return [(f"N{i}", f"N{(i + 1) % n}", 500 + i, f"2026-01-{1 + i % 28:02d} 12:00", "網路銀行", f"DN{i}") for i in range(n)]


def test_cash_rule_detects_atm_mule():
    rows = [("V1", "M", 60000, "2026-01-05 10:00", "網路銀行", "DV1")]
    rows += [("M", "CASH", 20000, f"2026-01-05 12:{m:02d}", "ATM提款", "") for m in (0, 5, 10)]
    ds = _txt(rows + _noise())
    assert "CASH" not in set(ds.accounts["account_id"])
    r, _ = run_rules(ds.transactions, ds.accounts, CFG["rules"])
    assert bool(r.loc["M", "R6_hit"]) and r.loc["M", "R6_max_count"] == 3
    assert bool(r.loc["M", "R1_hit"]) is False or r.loc["M", "R1_events"] >= 1


def test_device_rule_a1g():
    rows = [(a, "Z", 10000, f"2026-01-0{i + 2} 09:00", "網路銀行", "DX") for i, a in enumerate(["A", "B", "C"])]
    ds = _txt(rows + _noise())
    r, _ = run_rules(ds.transactions, ds.accounts, CFG["rules"])
    assert all(bool(r.loc[a, "R7_hit"]) for a in "ABC")
    assert not bool(r.loc["N1", "R7_hit"])


def test_tiny_rule():
    rows = [("T", "U", 1, f"2026-01-03 1{h}:00", "網路銀行", "DT") for h in range(3)]
    ds = _txt(rows + _noise())
    r, _ = run_rules(ds.transactions, ds.accounts, CFG["rules"])
    assert bool(r.loc["T", "R8_hit"])


def test_simulator_small(tmp_path):
    from flowaudit.simulator import generate
    small = {"months": 2, "n_office": 300, "n_student": 80, "n_retiree": 80, "n_self": 40, "n_landlord": 10,
             "n_groupbuy": 5, "n_merchant": 30, "n_employer": 8, "n_supplier": 5, "n_utility": 2, "n_reactivated": 5,
             "n_second": 60, "n_collector": 5, "n_family_mgr": 5, "n_seller": 10, "n_rosca": 5,
             "fraud_groups": {"假投資": 1, "網購詐騙": 1, "解除分期": 1, "循環交易": 1}}
    generate(tmp_path, cfg=small, verbose=False)
    ds = dl.load_twsim(tmp_path)
    assert ds.has_time and ds.has_alert_dates and ds.accounts["label"].sum() > 5
    assert (ds.transactions["dst"] == "CASH").any()
    r, _ = run_rules(ds.transactions, ds.accounts, CFG["rules"])
    assert r["rule_hits"].sum() > 0


def test_simulator_stress_knobs(tmp_path):
    from flowaudit.simulator import generate
    small = {"months": 2, "n_office": 200, "n_student": 50, "n_retiree": 50, "n_self": 20, "n_landlord": 5,
             "n_groupbuy": 3, "n_merchant": 20, "n_employer": 5, "n_supplier": 3, "n_utility": 2, "n_reactivated": 3,
             "n_second": 30, "n_collector": 3, "n_family_mgr": 3, "n_seller": 5, "n_rosca": 3,
             "fraud_groups": {"網購詐騙": 1}, "mule_source_p": [0.0, 0.0, 1.0], "p_test_tx": 0.0}
    _, acc, _ = generate(tmp_path, cfg=small, verbose=False)
    mules = acc[acc["label"] == 1]
    assert len(mules) > 0 and (mules["mule_source"] == "出售帳戶").all()


def test_cycle_fallback_excludes_ai_list():
    from flowaudit.evaluation import fallback_eval
    from flowaudit.model import cycle_fallback
    res = pd.DataFrame({"risk_rank": [1, 2, 3, 4, 5], "risk_score": [0.9, 0.8, 0.7, 0.6, 0.5],
                        "R4_cycles": [9, 0, 6, 2, 7], "label": [1, 0, 1, 0, 0],
                        "scheme": ["循環交易", "", "循環交易", "", ""]}, index=list("ABCDE"))
    q = cycle_fallback(res, top_k=2, min_cycles=5)
    assert list(q.index) == ["E", "C"]  # A 已在 AI 名單內；依循環次數排序
    fb = fallback_eval(res, top_k=2, ns=[5])
    assert fb["n_cycle_mules"] == 2 and fb["cycle_in_ai_top_k"] == 1
    assert fb["rows"][0] == {"min_cycles": 5, "n_queue": 2, "mules": 1, "precision": 0.5, "cycle_mules": 1}
    s = fb["system"]
    assert (s["ai_list_n"], s["rule_list_n"], s["fallback_n"], s["ai_only_mules"]) == (2, 0, 2, 1)
    assert s["ai_list_cycle"] + s["fallback_cycle"] == 2


def test_dual_list_reserves_rule_quota():
    from flowaudit.model import dual_list
    ai = np.array([0.9, 0.8, 0.7, 0.6, 0.1, 0.05])
    rule = np.array([0.0, 0.0, 0.0, 0.0, 3.0, 5.0])
    pick = dual_list(ai, rule, k=5, rule_quota=0.4)
    assert list(pick) == [0, 1, 2, 5, 4]  # AI 前 3 名＋不在 AI 名單中規則分數最高的 2 個
    assert list(dual_list(ai, rule, k=4, rule_quota=0.0)) == [0, 1, 2, 3]


def test_temporal_cycle_requires_time_order():
    ok = _tx([("A", "B", 100, 1), ("B", "C", 90, 2), ("C", "A", 80, 3)])
    assert len(find_temporal_cycles(ok.transactions)) == 1
    # 時間順序與資金方向相反（每一筆都早於前一筆），不構成資金循環
    bad = _tx([("A", "B", 100, 3), ("B", "C", 90, 2), ("C", "A", 80, 1)])
    assert len(find_temporal_cycles(bad.transactions)) == 0
    too_long = _tx([("A", "B", 100, 1), ("B", "C", 90, 2), ("C", "A", 80, 60)])
    assert len(find_temporal_cycles(too_long.transactions, max_span=30)) == 0


def test_pass_through_rule():
    rows = [("S", "M", 1000, 1), ("M", "X", 950, 2), ("S", "M", 1000, 10), ("M", "Y", 900, 11)]
    rows += [(f"N{i}", f"N{i+1}", 50, i) for i in range(20)]
    ds = _tx(rows)
    r, _ = run_rules(ds.transactions, ds.accounts, CFG["rules"])
    assert bool(r.loc["M", "R1_hit"])
    assert "快進快出" in r.loc["M", "rule_list"]


def test_fan_in_rule():
    rows = [(f"V{i}", "HUB", 100, 3) for i in range(15)]
    rows += [(f"N{i}", f"N{(i + 1) % 200}", 50, i % 30) for i in range(400)]
    ds = _tx(rows)
    r, _ = run_rules(ds.transactions, ds.accounts, CFG["rules"])
    assert bool(r.loc["HUB", "R2_hit"])
    assert r.loc["HUB", "R2_max_cp"] == 15


def test_amlworld_loader(tmp_path):
    csv = (
        "Timestamp,From Bank,Account,To Bank,Account,Amount Received,Receiving Currency,Amount Paid,Payment Currency,Payment Format,Is Laundering\n"
        "2022/09/01 00:20,10,8000EBD30,10,8000EBD30,3697.34,US Dollar,3697.34,US Dollar,Reinvestment,0\n"
        "2022/09/01 00:20,3208,8000F4580,1,8000F5340,0.01,US Dollar,0.01,US Dollar,Cheque,0\n"
        "2022/09/02 00:00,3209,8000F4670,3209,8000F4670,14675.57,US Dollar,14675.57,US Dollar,Reinvestment,0\n"
        "2022/09/03 00:02,12,8000F5030,12,8000F5040,2806.97,US Dollar,2806.97,US Dollar,Cheque,1\n"
        "2022/09/03 00:06,10,8000F5200,10,8000F5210,36682.97,Euro,36682.97,Euro,Cheque,0\n"
    )
    p = tmp_path / "HI-Tiny_Trans.csv"
    p.write_text(csv)
    ds = dl.load_amlworld(p)
    assert len(ds.transactions) == 2  # 排除自轉與歐元交易
    acc = ds.accounts.set_index("account_id")
    assert acc.loc["12_8000F5030", "label"] == 1 and acc.loc["3208_8000F4580", "label"] == 0
    assert ds.transactions["step"].tolist() == [0, 2]


def test_generic_csv_and_scoring():
    run = _run_or_skip("tw_sim")
    from flowaudit.model import load_model
    model = load_model(run["model_path"])
    tx = run["tx"].head(3000)[["src", "dst", "amount", "date", "channel", "device_id"]]
    ds = dl.load_generic_csv(io.StringIO(tx.to_csv(index=False)))
    out = analyze(ds, CFG, model=model, log=lambda *_: None)
    assert out["metrics"] is None
    assert out["results"]["risk_score"].between(0, 1).all()


def test_reports():
    run = _run_or_skip("tw_sim")
    res = run["results"]
    acct = res.sort_values("risk_rank").index[0]
    f = collect_facts(acct, res, run["shap"], run["X"], run["tx"])
    md, how = generate_report(f, {"provider": "template"})
    assert how == "template" and acct in md and "聲明" in md
    assert markdown_to_docx(md)[:2] == b"PK"
    assert build_workpaper(res, run["tx"], run["metrics"], run["summary"], 50)[:2] == b"PK"
    wb = pd.ExcelFile(io.BytesIO(build_workpaper(res, run["tx"], run["metrics"], run["summary"], 50, fallback_min=5,
                                                 rule_quota=0.1)))
    assert {"規則保底名單(資金循環)", "規則名單(新手法保險)"} <= set(wb.sheet_names)
    assert len(wb.parse("高風險帳戶清單")) == 45 and len(wb.parse("規則名單(新手法保險)")) == 5
    from flowaudit.model import review_lists
    ai_ids, rule_ids = review_lists(res, 50, 0.1)
    assert not set(ai_ids) & set(rule_ids) and len(ai_ids) + len(rule_ids) == 50


@pytest.mark.parametrize("key", ["tw_sim", "amlsim_fanin_cycle"])
def test_metrics_beat_baselines(key):
    m = _run_or_skip(key)["metrics"]
    assert m["model"]["pr_auc"] > m["rules"]["pr_auc"] > m["random_sampling"]["pr_auc"]


def test_evaluation_outputs():
    import json
    from flowaudit.pipeline import OUT_DIR
    ev = json.loads((OUT_DIR / "tw_sim" / "evaluation.json").read_text(encoding="utf-8"))
    s = ev["temporal"]["summary"]["100"]
    assert s["ai"]["unique_mules"] > s["rules"]["unique_mules"] > s["random"]["unique_mules"]
    assert ev["models"]["XGBoost（FlowAudit）"]["pr_auc"] > ev["models"]["現行規則計分"]["pr_auc"]
    for k in ("規則加權（依警示資料學權重）", "XGBoost（只用 16 項規則指標）"):
        assert k in ev["models"]
    assert ev["label_scarcity"]["1"]["n_train_positive"] == ev["n_train_positive"]
    assert "tw_sim" in ev["fallback"] and ev["fallback"]["tw_sim"]["rows"]
    assert {"dual", "cold", "cold_shift"} <= set(s)
    assert set(ev["unseen"]["schemes"]) == {"假投資", "網購詐騙", "解除分期"}


def test_tw_report_mentions_reference():
    run = _run_or_skip("tw_sim")
    res = run["results"]
    acct = res[res["rule_hits"] > 0].sort_values("risk_rank").index[0]
    f = collect_facts(acct, res, run["shap"], run["X"], run["tx"])
    md, _ = generate_report(f, {"provider": "template"})
    assert "態樣" in md or "管理辦法" in md
