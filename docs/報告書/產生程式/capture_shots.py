"""擷取儀表板畫面（報告書截圖原檔）。

需要：pip install playwright（使用本機的 Microsoft Edge，不需另外下載瀏覽器），
並先啟動儀表板：python -m streamlit run app.py（http://localhost:8501）。
畫面寬 1000 px、2 倍解析度、隱藏側邊欄，存到「截圖原檔」；裁切由 make_assets.py 處理。
"""
import time
from pathlib import Path

import numpy as np
from PIL import Image
from playwright.sync_api import sync_playwright

URL = "http://localhost:8501"
OUT = Path(__file__).parent / "截圖原檔"
OUT.mkdir(exist_ok=True)
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


def shot(pg, name):
    tag = pg.add_style_tag(content=HIDE)
    time.sleep(1.5)
    path = OUT / f"{name}.png"
    pg.screenshot(path=str(path), full_page=True)
    tag.evaluate("e => e.remove()")
    # 去掉下方空白
    im = Image.open(path).convert("RGB")
    rows = np.where((np.asarray(im.convert("L")) < 245).sum(axis=1) > 5)[0]
    im.crop((0, 0, im.width, min(int(rows.max()) + 60, im.height))).save(path, optimize=True)
    print("saved", path.name)


with sync_playwright() as p:
    b = p.chromium.launch(channel="msedge", headless=True)
    pg = b.new_page(viewport={"width": 1000, "height": 3400}, device_scale_factor=2)
    pg.goto(URL)
    pg.get_by_text("總覽：全查成果").wait_for(timeout=120000)
    settle(pg, 4)
    shot(pg, "1_總覽")
    goto(pg, "高風險帳戶清單")
    shot(pg, "2_高風險帳戶清單")
    goto(pg, "帳戶調查")
    pg.locator('[data-testid="stSelectbox"]').first.click()
    pg.keyboard.type("A017678")  # 命中 5 條規則、屬於疑似集團的帳戶，較能展示證據
    time.sleep(1)
    pg.keyboard.press("Enter")
    settle(pg, 4)
    shot(pg, "3_帳戶調查")
    goto(pg, "疑似集團分析")
    shot(pg, "4_疑似集團分析")
    b.close()
