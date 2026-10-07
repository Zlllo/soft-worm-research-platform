"""Shared Chinese font selection for plots and exported animations."""

import matplotlib
import matplotlib.font_manager as fm


CHINESE_FONT_CANDIDATES = (
    "PingFang SC", "Hiragino Sans GB", "Microsoft YaHei", "SimHei",
    "Noto Sans CJK SC", "Source Han Sans SC", "Arial Unicode MS",
    "Heiti SC", "STHeiti", "Songti SC",
)


def setup_chinese_font():
    """Select an installed font with Chinese glyphs without later overrides."""
    matplotlib.rcParams["axes.unicode_minus"] = False
    available = {font.name for font in fm.fontManager.ttflist}
    for name in CHINESE_FONT_CANDIDATES:
        if name not in available:
            continue
        try:
            path = fm.findfont(fm.FontProperties(family=[name]), fallback_to_default=False)
            charmap = fm.get_font(path).get_charmap()
            if not all(ord(char) in charmap for char in "线虫训练温度身体轮次奖励"):
                continue
        except (OSError, ValueError, RuntimeError):
            continue
        matplotlib.rcParams["font.family"] = ["sans-serif"]
        matplotlib.rcParams["font.sans-serif"] = [name, "DejaVu Sans"]
        return True
    matplotlib.rcParams["font.family"] = ["sans-serif"]
    matplotlib.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    return False
