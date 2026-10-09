import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def fonts(monkeypatch):
    matplotlib = SimpleNamespace(rcParams={})
    fm = SimpleNamespace(
        fontManager=SimpleNamespace(ttflist=[]),
        FontProperties=lambda **kwargs: kwargs,
        findfont=lambda props, **kwargs: props["family"][0],
        get_font=lambda path: SimpleNamespace(
            get_charmap=lambda: {ord(char): 1 for char in "线虫训练温度身体轮次奖励"}
        ),
    )
    matplotlib.font_manager = fm
    monkeypatch.setitem(sys.modules, "matplotlib", matplotlib)
    monkeypatch.setitem(sys.modules, "matplotlib.font_manager", fm)
    path = Path(__file__).resolve().parents[1] / "程序/core/plot_fonts.py"
    spec = importlib.util.spec_from_file_location("plot_fonts_under_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module, matplotlib, fm


def test_selects_installed_chinese_font_and_replaces_stale_setting(fonts):
    module, matplotlib, fm = fonts
    matplotlib.rcParams["font.sans-serif"] = ["SimHei"]
    fm.fontManager.ttflist = [SimpleNamespace(name="Hiragino Sans GB")]
    assert module.setup_chinese_font()
    assert matplotlib.rcParams["font.sans-serif"] == ["Hiragino Sans GB", "DejaVu Sans"]
    assert matplotlib.rcParams["axes.unicode_minus"] is False


def test_rejects_font_without_chinese_glyphs(fonts):
    module, matplotlib, fm = fonts
    fm.fontManager.ttflist = [SimpleNamespace(name="PingFang SC")]
    fm.get_font = lambda path: SimpleNamespace(get_charmap=lambda: {})
    assert not module.setup_chinese_font()
    assert matplotlib.rcParams["font.sans-serif"] == ["DejaVu Sans"]


def test_missing_fonts_use_available_default(fonts):
    module, matplotlib, _ = fonts
    assert not module.setup_chinese_font()
    assert matplotlib.rcParams["font.family"] == ["sans-serif"]


def test_unreadable_font_tries_next_candidate(fonts):
    module, matplotlib, fm = fonts
    fm.fontManager.ttflist = [SimpleNamespace(name=name) for name in ("PingFang SC", "Hiragino Sans GB")]
    original = fm.get_font

    def get_font(path):
        if path == "PingFang SC":
            raise OSError("unreadable font")
        return original(path)

    fm.get_font = get_font
    assert module.setup_chinese_font()
    assert matplotlib.rcParams["font.sans-serif"][0] == "Hiragino Sans GB"
