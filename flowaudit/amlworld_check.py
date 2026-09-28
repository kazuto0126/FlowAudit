"""IBM AMLworld 外部驗證（一鍵執行，不需要 AI 工具或帳號）。

用法
    python -m flowaudit.amlworld_check                     # 讀 data/raw/HI-Small_Trans.csv：先跑前 100 萬列，再跑全部
    python -m flowaudit.amlworld_check --stages 1000000    # 只跑前 100 萬列
    python -m flowaudit.amlworld_check --file 其他路徑.csv

結果存在 outputs/amlworld_check/：report.md（給人看）與 result_*.json（給程式讀），把整個資料夾交回即可。
每個階段在獨立的子程序執行：某階段記憶體不足或逾時，只影響該階段，前面的結果會保留。
AMLworld 沒有警示日期與集團資訊，因此以全部標籤訓練、一般分層交叉驗證評估（與台灣情境仿真資料的集團層級切分不同）。
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DEFAULT_FILE = ROOT / "data" / "raw" / "HI-Small_Trans.csv"
DEFAULT_OUT = ROOT / "outputs" / "amlworld_check"
KAGGLE_URL = "https://www.kaggle.com/datasets/ealtman2019/ibm-transactions-for-anti-money-laundering-aml"


def peak_memory_gb() -> float | None:
    """本程序到目前為止的記憶體峰值（GB）；取不到時回傳 None。"""
    try:
        if sys.platform == "win32":
            import ctypes
            from ctypes import wintypes

            class PMC(ctypes.Structure):
                _fields_ = [("cb", wintypes.DWORD), ("PageFaultCount", wintypes.DWORD)] + [
                    (n, ctypes.c_size_t) for n in ("PeakWorkingSetSize", "WorkingSetSize", "QuotaPeakPagedPoolUsage",
                                                   "QuotaPagedPoolUsage", "QuotaPeakNonPagedPoolUsage",
                                                   "QuotaNonPagedPoolUsage", "PagefileUsage", "PeakPagefileUsage")]

            k32, psapi = ctypes.WinDLL("kernel32"), ctypes.WinDLL("psapi")
            k32.GetCurrentProcess.restype = wintypes.HANDLE  # 64 位元的程序代號，不能用預設的 32 位元整數傳遞
            psapi.GetProcessMemoryInfo.argtypes = [wintypes.HANDLE, ctypes.POINTER(PMC), wintypes.DWORD]
            pmc = PMC()
            pmc.cb = ctypes.sizeof(PMC)
            ok = psapi.GetProcessMemoryInfo(k32.GetCurrentProcess(), ctypes.byref(pmc), pmc.cb)
            return pmc.PeakWorkingSetSize / 1e9 if ok else None
        import resource
        r = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return r / 1e9 if sys.platform == "darwin" else r * 1024 / 1e9  # macOS 單位為位元組、Linux 為 KB
    except Exception:
        return None


def total_memory_gb() -> float | None:
    try:
        if sys.platform == "win32":
            import ctypes

            class MS(ctypes.Structure):
                _fields_ = [("dwLength", ctypes.c_ulong), ("dwMemoryLoad", ctypes.c_ulong)] + [
                    (n, ctypes.c_ulonglong) for n in ("ullTotalPhys", "ullAvailPhys", "ullTotalPageFile", "ullAvailPageFile",
                                                      "ullTotalVirtual", "ullAvailVirtual", "ullAvailExtendedVirtual")]

            ms = MS()
            ms.dwLength = ctypes.sizeof(MS)
            ctypes.windll.kernel32.GlobalMemoryStatusEx(ctypes.byref(ms))
            return ms.ullTotalPhys / 1e9
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1e9
    except Exception:
        return None


def environment() -> dict:
    from importlib.metadata import PackageNotFoundError, version

    def ver(p):
        try:
            return version(p)
        except PackageNotFoundError:
            return "未安裝"

    try:
        commit = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=ROOT, capture_output=True, text=True,
                                timeout=10).stdout.strip() or "未知"
    except Exception:
        commit = "未知（非 git 下載）"
    mem = total_memory_gb()
    return {"os": f"{platform.system()} {platform.release()}", "cpu_count": os.cpu_count(),
            "memory_gb": round(mem, 1) if mem else None, "python": platform.python_version(), "commit": commit,
            **{p: ver(p) for p in ("xgboost", "scikit-learn", "pandas", "numpy", "networkx")}}


# ---------------------------------------------------------------------------
def run_stage(path: Path, nrows: int | None, out_json: Path):
    """子程序：跑一個階段，結果寫成 JSON。"""
    from . import data_loader as dl
    from .pipeline import analyze, load_config

    t0 = time.time()
    marks = []

    def log(msg):
        marks.append((msg, time.time() - t0))
        print(f"  [{time.time() - t0:>6.0f}s｜記憶體峰值 {peak_memory_gb() or 0:.1f} GB] {msg}", flush=True)

    ds = dl.load_amlworld(path, nrows=nrows)
    log(f"讀取完成：{len(ds.transactions):,} 筆交易、{len(ds.accounts):,} 個帳戶")
    cfg = load_config()
    run = analyze(ds, cfg, log=log)
    total = time.time() - t0
    steps = [{"step": m, "sec": round((marks[i + 1][1] if i + 1 < len(marks) else total) - s, 1)}
             for i, (m, s) in enumerate(marks)]
    m = run["metrics"]
    max_cycles = 20000  # 與 rules.find_temporal_cycles 的預設上限相同
    out = {
        "nrows": nrows, "summary": ds.summary(), "elapsed_sec": round(total, 1), "steps": steps,
        "peak_memory_gb": round(peak_memory_gb() or 0, 2),
        "n_cycles_found": len(run["cycles"]), "cycle_search_capped": len(run["cycles"]) >= max_cycles,
        "metrics": {k: m[k] for k in ("positive_rate", "model", "rules", "random_sampling", "train_label") if k in m},
    }
    Path(out_json).write_text(json.dumps(out, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    print(f"  完成：AI PR-AUC {m['model']['pr_auc']:.3f}｜規則 {m['rules']['pr_auc']:.3f}｜隨機 {m['positive_rate']:.4f}"
          f"｜{total / 60:.1f} 分鐘｜記憶體峰值 {out['peak_memory_gb']:.1f} GB", flush=True)


def _stage_name(nrows):
    return f"前 {nrows:,} 列" if nrows else "全部資料"


def write_report(records: list, env: dict, path: Path, out_dir: Path):
    L = ["# IBM AMLworld 外部驗證結果", "",
         f"- 資料檔：`{path.name}`（只保留美元交易，避免不同幣別金額混算）",
         f"- 程式版本：git commit `{env['commit']}`",
         f"- 電腦：{env['os']}，{env['cpu_count']} 個 CPU 核心，記憶體 {env['memory_gb']} GB",
         f"- 套件：Python {env['python']}、xgboost {env['xgboost']}、scikit-learn {env['scikit-learn']}、"
         f"pandas {env['pandas']}、numpy {env['numpy']}、networkx {env['networkx']}", "",
         "## 各階段", "",
         "| 階段 | 狀態 | 交易筆數 | 帳戶數 | 洗錢帳戶比例 | AI PR-AUC | 規則 PR-AUC | 總耗時 | 記憶體峰值 |",
         "|---|---|---|---|---|---|---|---|---|"]
    for r in records:
        x = r.get("result")
        if x:
            s, m = x["summary"], x["metrics"]
            L.append(f"| {_stage_name(r['nrows'])} | {r['status']} | {s['n_transactions']:,} | {s['n_accounts']:,} | "
                     f"{m['positive_rate']:.2%} | **{m['model']['pr_auc']:.3f}** | {m['rules']['pr_auc']:.3f} | "
                     f"{x['elapsed_sec'] / 60:.1f} 分鐘 | {x['peak_memory_gb']:.1f} GB |")
        else:
            L.append(f"| {_stage_name(r['nrows'])} | {r['status']} | — | — | — | — | — | {r['wall_sec'] / 60:.1f} 分鐘 | — |")
    for r in records:
        x = r.get("result")
        if not x:
            continue
        m = x["metrics"]
        L += ["", f"## {_stage_name(r['nrows'])}：覆核前 K 名的命中率", "",
              "| 覆核帳戶數 | AI 模型 | 傳統規則 | 隨機抽樣 |", "|---|---|---|---|"]
        for a, b in zip(m["model"]["at_k"], m["rules"]["at_k"]):
            L.append(f"| {a['k']:,} | {a['precision']:.1%} | {b['precision']:.1%} | {m['positive_rate']:.2%} |")
        L += ["", "各步驟耗時：" + "；".join(f"{s['step'].split('…')[0].strip()} {s['sec']:.0f} 秒" for s in x["steps"]),
              "", f"資金循環搜尋找到 {x['n_cycles_found']:,} 個循環"
              + ("（達到 20,000 個上限後停止搜尋）" if x["cycle_search_capped"] else "") + "。"]
    L += ["", "> 說明：PR-AUC 為 5 折交叉驗證的樣本外結果（每個帳戶的分數都來自沒看過它的模型）。",
          "> AMLworld 沒有警示日期與詐騙集團資訊，因此以全部標籤訓練、一般分層交叉驗證評估，",
          "> 與台灣情境仿真資料「只用已警示帳戶訓練、集團層級切分」的設定不同，兩者數字不能直接比較。"]
    (out_dir / "report.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    (out_dir / "environment.json").write_text(json.dumps(env, ensure_ascii=False, indent=2), encoding="utf-8")


def main():
    ap = argparse.ArgumentParser(description="FlowAudit：IBM AMLworld 外部驗證（一鍵執行）")
    ap.add_argument("--file", type=Path, default=DEFAULT_FILE, help="AMLworld 交易檔（預設 data/raw/HI-Small_Trans.csv）")
    ap.add_argument("--stages", nargs="+", default=["1000000", "all"], help="依序執行的階段：讀取列數或 all")
    ap.add_argument("--out", type=Path, default=DEFAULT_OUT, help="結果資料夾")
    ap.add_argument("--timeout-hours", type=float, default=3.0, help="每個階段最多執行幾小時")
    ap.add_argument("--worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--nrows", type=int, default=None, help=argparse.SUPPRESS)
    ap.add_argument("--out-json", type=Path, default=None, help=argparse.SUPPRESS)
    args = ap.parse_args()
    if args.worker:
        run_stage(args.file, args.nrows, args.out_json)
        return
    if not args.file.exists():
        print(f"找不到資料檔：{args.file}\n請先從 Kaggle 下載 HI-Small_Trans.csv，放到 data/raw/ 資料夾：\n{KAGGLE_URL}")
        sys.exit(1)
    args.out.mkdir(parents=True, exist_ok=True)
    env = environment()
    print(f"電腦：{env['os']}，{env['cpu_count']} 核心，記憶體 {env['memory_gb']} GB；Python {env['python']}", flush=True)
    records = []
    for st in args.stages:
        nrows = None if st.lower() == "all" else int(st)
        out_json = args.out / f"result_{'all' if nrows is None else nrows}.json"
        out_json.unlink(missing_ok=True)
        print(f"\n== {_stage_name(nrows)}（最多 {args.timeout_hours:g} 小時）==", flush=True)
        cmd = [sys.executable, "-m", "flowaudit.amlworld_check", "--worker", "--file", str(args.file),
               "--out-json", str(out_json)] + (["--nrows", str(nrows)] if nrows else [])
        t0 = time.time()
        try:
            rc = subprocess.run(cmd, cwd=ROOT, timeout=args.timeout_hours * 3600).returncode
            status = "完成" if rc == 0 and out_json.exists() else f"失敗（結束代碼 {rc}，常見原因是記憶體不足）"
        except subprocess.TimeoutExpired:
            status = f"逾時（超過 {args.timeout_hours:g} 小時）"
        rec = {"nrows": nrows, "status": status, "wall_sec": round(time.time() - t0, 1),
               "result": json.loads(out_json.read_text(encoding="utf-8")) if status == "完成" else None}
        records.append(rec)
        write_report(records, env, args.file, args.out)
        if status != "完成":
            print(f"  {status}；後面更大的階段不再執行。", flush=True)
            break
    print(f"\n報告：{args.out / 'report.md'}\n請把整個 {args.out} 資料夾傳回。", flush=True)


if __name__ == "__main__":
    main()
