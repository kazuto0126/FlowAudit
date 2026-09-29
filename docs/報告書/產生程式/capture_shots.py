"""擷取儀表板畫面。

    python capture_shots.py          # 報告書與海報用的截圖原檔（寬 1000 px、2 倍解析度、隱藏側邊欄）→ 截圖原檔/
    python capture_shots.py readme   # README 用的 12 張全頁截圖（寬 1500 px、含側邊欄）→ docs/screenshots/

需要：pip install playwright（使用本機的 Microsoft Edge，不需另外下載瀏覽器），
並先啟動儀表板：python -m streamlit run app.py（http://localhost:8501）。裁切由 make_assets.py 處理。
"""
import sys
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
HERE = Path(__file__).parent
HIDE = '[data-testid="stSidebar"]{display:none !important}'


def settle(pg, extra=2.5):
    """等 Streamlit 跑完（右上角的執行中圖示消失），再等圖表繪製。"""
    time.sleep(1.0)
    for _ in range(120):
        if pg.locator('[data-testid="stStatusWidget"]').count() == 0:
            break
        time.sleep(0.5)
    time.sleep(extra)


def goto(pg, name):
    pg.locator('[data-testid="stSidebar"]').get_by_text(name, exact=True).click()
    settle(pg)


def pick_account(pg, account="A017678"):
    """帳戶調查頁選一個命中 5 條規則、屬於疑似集團的帳戶，較能展示證據。"""
    pg.locator('[data-testid="stSelectbox"]').first.click()
    pg.keyboard.type(account)
    time.sleep(1)
    pg.keyboard.press("Enter")
    settle(pg, 4)


def trim(path, x0=0, max_h=None, pad=60):
    """去掉下方空白（只看 x0 右邊的主畫面，側邊欄有底色）。"""
    im = Image.open(path).convert("RGB")
    a = np.asarray(im.convert("L"))[:, x0:]
    rows = np.where((a < 245).sum(axis=1) > 5)[0]
    h = min(int(rows.max()) + pad, im.height, max_h or im.height)
    im.crop((0, 0, im.width, h)).save(path, optimize=True)


def report_shots(pg):
    out = HERE / "截圖原檔"
    out.mkdir(exist_ok=True)

    def shot(name):
        tag = pg.add_style_tag(content=HIDE)
        time.sleep(1.5)
        pg.screenshot(path=str(out / f"{name}.png"), full_page=True)
        tag.evaluate("e => e.remove()")
        trim(out / f"{name}.png")
        print("saved", name)

    shot("1_總覽")
    goto(pg, "高風險帳戶清單")
    shot("2_高風險帳戶清單")
    goto(pg, "帳戶調查")
    pick_account(pg)
    shot("3_帳戶調查")
    goto(pg, "疑似集團分析")
    shot("4_疑似集團分析")


def readme_shots(pg):
    out = HERE.parents[1] / "screenshots"

    def shot(name, full=True):
        """Streamlit 的內容在內層捲動，所以用很高的視窗一次拍完整頁；full=False 只拍第一個畫面。"""
        path = out / f"{name}.png"
        if full:
            pg.screenshot(path=str(path))
            trim(path, x0=320, max_h=3000)
        else:
            pg.screenshot(path=str(path), clip={"x": 0, "y": 0, "width": 1500, "height": 1307})
        print("saved", name)

    shot("1_總覽")
    goto(pg, "高風險帳戶清單")
    shot("2_高風險帳戶清單")
    goto(pg, "帳戶調查")
    pick_account(pg)
    shot("3_帳戶調查")
    goto(pg, "疑似集團分析")
    shot("4_疑似集團分析")
    goto(pg, "模型評估")
    settle(pg, 3)
    shot("5_模型評估", full=False)
    for name, tab in (("5a_逐月持續稽核模擬", "逐月持續稽核模擬"), ("5b_成本效益試算", "成本效益試算"),
                      ("5c_多模型比較", "多模型比較"), ("5d_消融實驗", "消融實驗"), ("5e_誤判分析", "誤判分析")):
        pg.get_by_role("tab", name=tab).click()
        settle(pg, 3)
        shot(name)
    goto(pg, "上傳新資料全查")
    shot("6_上傳新資料全查")
    goto(pg, "系統架構、資料與設定")
    shot("7_系統架構、資料與設定")


if __name__ == "__main__":
    readme = len(sys.argv) > 1 and sys.argv[1] == "readme"
    with sync_playwright() as p:
        b = p.chromium.launch(channel="msedge", headless=True)
        if readme:
            pg = b.new_page(viewport={"width": 1500, "height": 3000}, device_scale_factor=1)
        else:
            pg = b.new_page(viewport={"width": 1000, "height": 3400}, device_scale_factor=2)
        pg.goto(URL)
        pg.get_by_text("總覽：全查成果").wait_for(timeout=120000)
        settle(pg, 4)
        (readme_shots if readme else report_shots)(pg)
        b.close()
