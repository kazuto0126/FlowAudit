"""報告書用的圖表與截圖裁切。數字一律從 outputs/tw_sim/evaluation.json 等結果檔讀取，不手動抄寫。

截圖原檔在「截圖原檔」資料夾，由 capture_shots.py 從儀表板擷取（寬 1000 px、2 倍解析度、隱藏側邊欄）。
"""
import json
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import font_manager
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image, ImageDraw

HERE = Path(__file__).parent
ROOT = HERE.parents[2]
OUT = HERE / "assets"
SHOTS = HERE / "截圖原檔"
OUT.mkdir(exist_ok=True)
sys.path.insert(0, str(ROOT))

# 圖表顏色沿用 app.py（CLAUDE.md 規定不另加顏色）
C_AI, C_RULE, C_RAND = "#2a78d6", "#eb6834", "#1baf7a"
C_GRAY = "#9a9892"
INK, SUB, LINE = "#1f1f1f", "#52514e", "#c9c8c4"

font_manager.fontManager.addfont(r"C:\Windows\Fonts\msjh.ttc")
font_manager.fontManager.addfont(r"C:\Windows\Fonts\msjhbd.ttc")
plt.rcParams.update({"font.family": "Microsoft JhengHei", "font.size": 10, "axes.unicode_minus": False,
                     "axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": LINE,
                     "axes.labelcolor": SUB, "xtick.color": SUB, "ytick.color": SUB, "savefig.dpi": 300})
CM = 1 / 2.54
ev = json.loads((ROOT / "outputs/tw_sim/evaluation.json").read_text(encoding="utf-8"))


def save(fig, name):
    """報告書用 PNG；海報另用 SVG（向量，放大列印不失真）。"""
    fig.savefig(OUT / f"{name}.png", facecolor="white")
    fig.savefig(OUT / f"{name}.svg", facecolor="white")


def canvas(w_cm, h_cm):
    """以公分為座標的畫布：字級（pt）與方框（cm）的比例固定，不會因縮放而擠出方框。"""
    fig = plt.figure(figsize=(w_cm * CM, h_cm * CM))
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, w_cm); ax.set_ylim(0, h_cm); ax.axis("off")
    return fig, ax


def box(ax, x, y, w, h, fc="#ffffff", ec=LINE, lw=1.1, ls="-"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0,rounding_size=0.15", fc=fc, ec=ec, lw=lw,
                                linestyle=ls, clip_on=False))


def text(ax, x, y, s, size=8, bold=False, color=INK, ha="center", va="center"):
    ax.text(x, y, s, ha=ha, va=va, fontsize=size, color=color, fontweight="bold" if bold else "normal",
            linespacing=1.35, clip_on=False)


def arrow(ax, p, q, color=SUB, ls="-", rad=0.0, lw=1.1):
    ax.add_patch(FancyArrowPatch(p, q, arrowstyle="-|>", mutation_scale=10, color=color, lw=lw, linestyle=ls,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=1, shrinkB=1, clip_on=False))


# ---------------------------------------------------------------- 圖：系統架構（16 × 9.4 cm）
def architecture():
    fig, ax = canvas(16, 9.4)
    # 輸入資料
    box(ax, 0.1, 4.4, 2.6, 2.4, fc="#f4f4f2")
    text(ax, 1.4, 6.35, "交易資料", size=8.5, bold=True)
    text(ax, 1.4, 5.35, "時間、金額、管道\n登入裝置\n開戶日、休眠天數", size=7)
    # ①～④
    steps = [("①紅旗規則引擎", "8 條規則全查\n對應態樣與法規"), ("②特徵工程", "57 項特徵\n含資金流向圖"),
             ("③AI 風險模型", "XGBoost\n只用已警示帳戶訓練"), ("④可解釋分析", "TreeSHAP\n＋規則證據")]
    xs = [3.15, 6.35, 9.55, 12.75]
    for (t, d), x in zip(steps, xs):
        box(ax, x, 4.4, 2.8, 2.4, fc="#e9f1fb", ec=C_AI)
        text(ax, x + 1.4, 6.25, t, size=8.5, bold=True)
        text(ax, x + 1.4, 5.15, d, size=7, color=SUB)
    arrow(ax, (2.7, 5.6), (3.15, 5.6))
    for a, b in zip(xs, xs[1:]):
        arrow(ax, (a + 2.8, 5.6), (b, 5.6))
    # 冷啟動
    box(ax, 9.55, 7.7, 2.8, 1.0, ec=C_GRAY, ls="--")
    text(ax, 10.95, 8.2, "外部參考資料\n（冷啟動）", size=7, color=SUB)
    arrow(ax, (10.95, 7.7), (10.95, 6.8), color=C_GRAY, ls="--")
    # ⑤ 覆核名單
    box(ax, 3.15, 0.3, 8.8, 2.3, fc="#fdf1ea", ec=C_RULE)
    text(ax, 7.55, 2.2, "⑤覆核名單", size=8.5, bold=True)
    for i, (t, c) in enumerate([("AI 名單", C_AI), ("規則名單\n（新手法保險）", C_RULE), ("規則保底名單\n（資金循環）", C_RULE)]):
        box(ax, 3.4 + i * 2.85, 0.5, 2.6, 1.3, ec=c)
        text(ax, 4.7 + i * 2.85, 1.15, t, size=7.5)
    arrow(ax, (4.55, 4.4), (6.0, 2.6), color=C_RULE)
    text(ax, 5.45, 3.5, "規則分數", size=7, color=C_RULE, ha="left")
    arrow(ax, (10.2, 4.4), (8.4, 2.6), color=C_AI)
    text(ax, 9.05, 3.5, "風險分數", size=7, color=C_AI, ha="right")
    # 產出
    box(ax, 12.75, 0.3, 2.8, 2.3, fc="#f4f4f2")
    text(ax, 14.15, 2.2, "產出", size=8.5, bold=True)
    text(ax, 14.15, 1.15, "儀表板（7 頁）\n可疑交易分析報告\nExcel 稽核工作底稿\n疑似集團偵測", size=7)
    arrow(ax, (11.95, 1.45), (12.75, 1.45))
    arrow(ax, (14.15, 4.4), (14.15, 2.6))
    # 回饋
    arrow(ax, (11.4, 2.6), (11.4, 4.4), color=C_GRAY, ls="--")
    text(ax, 11.55, 3.5, "覆核結論回饋\n每月重新訓練", size=7, color=SUB, ha="left")
    save(fig, "fig_architecture")
    plt.close(fig)


# ---------------------------------------------------------------- 圖：每月作業流程（16 × 4.6 cm）
def workflow():
    fig, ax = canvas(16, 4.6)
    steps = [("月初全查", "規則＋AI 評分"), ("產生覆核名單", "AI＋規則＋保底"), ("稽核人員調查", "判斷依據\n資金網路圖"),
             ("覆核結論", "分析報告\n稽核工作底稿"), ("處置", "警示凍結\n疑似洗錢申報")]
    w, gap, x0 = 2.7, 0.45, 0.15
    for i, (t, d) in enumerate(steps):
        x = x0 + i * (w + gap)
        box(ax, x, 1.7, w, 2.1, fc="#e9f1fb" if i < 2 else "#ffffff", ec=C_AI)
        text(ax, x + w / 2, 3.35, t, size=8.5, bold=True)
        text(ax, x + w / 2, 2.4, d, size=7.5, color=SUB)
        if i:
            arrow(ax, (x - gap, 2.75), (x, 2.75))
    last = x0 + 4 * (w + gap) + w / 2
    arrow(ax, (last, 1.7), (x0 + w / 2, 1.7), rad=-0.12, ls="--", color=C_GRAY)
    text(ax, 8.0, 0.25, "確認的人頭帳戶加入訓練資料，下個月的模型更準（持續稽核）", size=7.5, color=SUB)
    save(fig, "fig_workflow")
    plt.close(fig)


# ---------------------------------------------------------------- 圖：相同覆核量下累計找到的人頭帳戶
def equal_volume():
    f = ev["fair"]
    cv = f["curve"]
    n_pos, n_acc = f["n_mules"], f["n_accounts"]
    at = next(r for r in f["by_k"] if r["k"] == n_pos)
    fig, ax = plt.subplots(figsize=(16 * CM, 6.6 * CM))
    ax.plot(cv["k"], cv["ai"], color=C_AI, lw=1.8, label="AI 模型（依風險分數）")
    ax.plot(cv["k"], cv["rules"], color=C_RULE, lw=1.8, label="比較基準：規則（依規則分數）")
    ax.plot(cv["k"], [k * n_pos / n_acc for k in cv["k"]], color=C_RAND, lw=1.5, label="隨機抽樣（期望值）")
    ax.axhline(n_pos, color=C_GRAY, lw=0.8, ls=":")
    ax.text(cv["k"][-1], n_pos + 4, f"人頭帳戶總數 {n_pos}", ha="right", va="bottom", fontsize=7.5, color=SUB)
    ax.axvline(n_pos, color=C_GRAY, lw=0.8, ls="--")
    for v, c, lab, dy in ((at["ai"]["mules"], C_AI, "AI", -22), (at["rules"]["mules"], C_RULE, "規則", -18),
                          (at["random_mules"], C_RAND, "隨機", 14)):
        ax.plot([n_pos], [v], "o", color=c, ms=4)
        ax.text(n_pos + 14, v + dy, f"{lab} {v:.0f} 個", va="center", fontsize=8, color=c, fontweight="bold",
                bbox=dict(fc="white", ec="none", pad=0.5))
    ax.text(n_pos - 8, 12, f"覆核 {n_pos} 個帳戶", ha="right", fontsize=7.5, color=SUB)
    ax.set_xlim(0, cv["k"][-1]); ax.set_ylim(0, n_pos * 1.12)
    ax.set_xlabel("人工覆核的帳戶數（依分數由高到低）", fontsize=8.5)
    ax.set_ylabel("累計找到的人頭帳戶數", fontsize=8.5)
    ax.tick_params(labelsize=8)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_equal_volume")
    plt.close(fig)


# ---------------------------------------------------------------- 圖：解決方案比較（逐月模擬，每月覆核 100 個）
def solutions():
    s = ev["temporal"]["summary"]["100"]
    items = [("傳統規則", "rules", C_RULE, None), ("AI（只用自家警示資料）", "ai", C_AI, None),
             ("雙名單（保留 10% 給規則）", "dual", C_AI, "//"),
             ("冷啟動：參考資料分布不同", "cold_shift", C_AI, ".."),
             ("冷啟動：參考資料分布相同", "cold", C_AI, "xx")]
    fig, axes = plt.subplots(1, 2, figsize=(16 * CM, 5.6 * CM), sharey=True)
    for ax, key, title, fmt in ((axes[0], "unique_mules", "6 個月找到的人頭帳戶（個）", "{:.0f}"),
                                (axes[1], "prevented_amount", "潛在攔阻金額（萬元，確認當天凍結）", "{:,.0f}")):
        vals = [s[k][key] / (1e4 if key == "prevented_amount" else 1) for _, k, _, _ in items]
        y = range(len(items))[::-1]
        for yi, v, (_, _, c, h) in zip(y, vals, items):
            ax.barh(yi, v, color=c if h is None else "white", edgecolor=c, hatch=h, lw=1.1, height=0.62)
            ax.text(v, yi, " " + fmt.format(v), va="center", fontsize=8.5, color=INK)
        ax.set_title(title, fontsize=9.5, color=INK, loc="left")
        ax.set_xlim(0, max(vals) * 1.22)
        ax.tick_params(axis="x", labelsize=8)
    axes[0].set_yticks(list(range(len(items)))[::-1], [n for n, _, _, _ in items], fontsize=8.5)
    fig.tight_layout(w_pad=1.5)
    save(fig, "fig_solutions")
    plt.close(fig)


# ---------------------------------------------------------------- 圖：穩健性測試與外部資料驗證
def external():
    m = ev["models"]
    ams = json.loads((HERE / "amlworld_all.json").read_text(encoding="utf-8"))["metrics"]
    metrics = json.loads((ROOT / "outputs/tw_sim/metrics.json").read_text(encoding="utf-8"))
    seed2 = metrics["external_validation"]["tw_sim_seed2"]
    groups = [("台灣情境仿真資料\n（集團層級交叉驗證）", m["XGBoost（FlowAudit）"]["pr_auc"], m["現行規則計分"]["pr_auc"],
               metrics["positive_rate"]),
              ("另一份仿真資料\n（模型不重新訓練）", seed2["model_pr_auc"], seed2["rules_pr_auc"], seed2["positive_rate"]),
              ("IBM AMLworld\n（以其標籤重新訓練、5 折）", ams["model"]["pr_auc"], ams["rules"]["pr_auc"], ams["positive_rate"])]
    fig, ax = plt.subplots(figsize=(16 * CM, 6.2 * CM))
    wbar = 0.26
    for i, (name, a, r, base) in enumerate(groups):
        for j, (v, c, lab) in enumerate(((a, C_AI, "AI 模型"), (r, C_RULE, "規則（比較基準）"), (base, C_RAND, "隨機抽樣"))):
            x = i + (j - 1) * wbar
            ax.bar(x, v, wbar * 0.92, color=c, label=lab if i == 0 else None)
            ax.text(x, v + 0.015, f"{v:.3f}", ha="center", fontsize=8, color=INK)
    ax.set_xticks(range(3), [g[0] for g in groups], fontsize=8.5)
    ax.set_ylim(0, 1.08); ax.set_ylabel("PR-AUC（越高越好）", fontsize=8.5)
    ax.tick_params(axis="y", labelsize=8)
    ax.legend(frameon=False, ncol=3, fontsize=8.5, loc="lower center", bbox_to_anchor=(0.5, 1.0))
    fig.tight_layout()
    save(fig, "fig_external")
    plt.close(fig)


# ---------------------------------------------------------------- 截圖裁切（座標為畫面 px，原檔為 2 倍解析度）
def crop(name, box_px):
    return Image.open(SHOTS / f"{name}.png").convert("RGB").crop(tuple(v * 2 for v in box_px))


def stack(parts, gap=16):
    w = max(p.width for p in parts)
    out = Image.new("RGB", (w, sum(p.height for p in parts) + gap * (len(parts) - 1)), "white")
    y = 0
    for p in parts:
        out.paste(p, (0, y))
        y += p.height + gap
    return out


def side(parts, gap=24):
    h = max(p.height for p in parts)
    out = Image.new("RGB", (sum(p.width for p in parts) + gap * (len(parts) - 1), h), "white")
    x = 0
    for p in parts:
        out.paste(p, (x, 0))
        x += p.width + gap
    return out


def crops():
    crop("1_總覽", (70, 95, 925, 430)).save(OUT / "shot_overview.png")
    # 清單：表格前 8 列＋另外兩份名單（不含右側被截斷的「AI 主要判斷依據」欄）
    stack([crop("2_高風險帳戶清單", (80, 340, 878, 665)), crop("2_高風險帳戶清單", (80, 958, 878, 1066))]).save(
        OUT / "shot_list.png")
    # 帳戶調查：AI 判斷依據與資金往來網路；左下方屬於下一段（命中的紅旗規則），以白色蓋掉
    top = crop("3_帳戶調查", (72, 448, 918, 1105))
    d = ImageDraw.Draw(top)
    d.rectangle((0, 1045, 860, top.height), fill="white")
    d.rectangle((640, 15, 720, 90), fill="white")  # 標題旁的連結圖示
    top.save(OUT / "shot_investigate.png")
    side([crop("3_帳戶調查", (72, 1036, 500, 1300)), crop("3_帳戶調查", (72, 1308, 500, 1505))]).save(
        OUT / "shot_evidence.png")
    crop("4_疑似集團分析", (78, 805, 922, 1407)).save(OUT / "shot_group.png")  # 上方有驗證說明框


if __name__ == "__main__":
    architecture(); workflow(); equal_volume(); solutions(); external(); crops()
    for p in sorted(OUT.glob("*.png")):
        w, h = Image.open(p).size
        print(f"{p.name}: {w}x{h}")
