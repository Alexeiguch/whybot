"""WCAG AA (4.5:1) for every text/background pair the playground CSS uses (playground/theme.py PAIRS)."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from playground import theme  # noqa: E402


def test_every_custom_colour_pair_meets_wcag_aa():
    bad = [(f, b, c) for f, b, c in theme.contrast_report() if c < 4.5]
    assert not bad, f"pairs below 4.5:1: {bad}"


def test_every_role_has_a_distinct_badge():
    bgs = [theme.PALETTE[f"{r}_bg"] for r in theme.ROLES]
    assert len(set(bgs)) == len(bgs)
    css = theme.css()
    assert all(f".ag-{r}{{" in css for r in theme.ROLES)


def test_theme_is_identical_in_light_and_dark():
    t = theme.build_theme()
    diff = [k for k in vars(t) if k.endswith("_dark") and getattr(t, k) is not None and getattr(t, k) != getattr(t, k[:-5])]
    assert not diff, diff
    assert "--body-background-fill" in t._get_theme_css()  # what Gradio does at launch
