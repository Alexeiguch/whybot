"""One fixed dark theme for the playground, identical in light and dark browser modes.

Every custom colour lives in PALETTE, every text/background combination the CSS uses is listed in PAIRS, and
playground/tests/test_contrast.py asserts WCAG AA (4.5:1) for each pair, so a colour change cannot silently break contrast.
"""
from __future__ import annotations

import gradio as gr

PALETTE = {
    # surfaces
    "page": "#0b1120", "panel": "#131c2e", "raised": "#1b2740", "code": "#070c17", "border": "#33415c",
    # text
    "text": "#f1f5f9", "text_muted": "#cbd5e1", "text_subtle": "#a3b1c6",
    "accent": "#7cc4ff", "on_accent": "#06121f",
    # banners (solid)
    "rec_bg": "#fbbf24", "rec_fg": "#1c1300",
    "live_bg": "#15803d", "live_fg": "#ffffff",
    "mixed_bg": "#1d4ed8", "mixed_fg": "#ffffff",
    "idle_bg": "#334155", "idle_fg": "#f8fafc",
    "warnline_fg": "#7a2e00",
    # verdict / status pills
    "ok_bg": "#15803d", "ok_fg": "#ffffff",
    "weak_bg": "#fbbf24", "weak_fg": "#1c1300",
    "bad_bg": "#b91c1c", "bad_fg": "#ffffff",
    "retry_bg": "#1d4ed8", "retry_fg": "#ffffff",
    "muted_bg": "#475569", "muted_fg": "#ffffff",
    # tinted rows / cards (text on them is `text`)
    "ok_tint": "#0f2a1c", "weak_tint": "#2e2408", "bad_tint": "#3a1015", "retry_tint": "#0f1f46",
    "ok_line": "#22c55e", "weak_line": "#fbbf24", "bad_line": "#f87171", "retry_line": "#60a5fa",
    "bad_text": "#fca5a5",
    # agent role badges (bg, fg)
    "Planner_bg": "#c4b5fd", "Planner_fg": "#1e1048",
    "Executor_bg": "#7dd3fc", "Executor_fg": "#062a40",
    "Analyst_bg": "#6ee7b7", "Analyst_fg": "#032b1f",
    "Critic_bg": "#fda4af", "Critic_fg": "#4a0613",
    "Writer_bg": "#fcd34d", "Writer_fg": "#3b1d00",
    "Memory_bg": "#f0abfc", "Memory_fg": "#3d0745",
    "Orchestrator_bg": "#e2e8f0", "Orchestrator_fg": "#0f172a",
}
ROLES = ("Planner", "Executor", "Analyst", "Critic", "Writer", "Memory", "Orchestrator")

_P = PALETTE
PAIRS: list[tuple[str, str]] = [
    ("text", "page"), ("text", "panel"), ("text", "raised"), ("text", "code"),
    ("text_muted", "page"), ("text_muted", "panel"), ("text_muted", "raised"), ("text_subtle", "panel"), ("text_subtle", "raised"),
    ("text_subtle", "page"), ("accent", "panel"), ("accent", "code"), ("on_accent", "accent"),
    ("rec_fg", "rec_bg"), ("live_fg", "live_bg"), ("mixed_fg", "mixed_bg"), ("idle_fg", "idle_bg"), ("warnline_fg", "rec_bg"),
    ("ok_fg", "ok_bg"), ("weak_fg", "weak_bg"), ("bad_fg", "bad_bg"), ("retry_fg", "retry_bg"), ("muted_fg", "muted_bg"),
    ("text", "ok_tint"), ("text", "weak_tint"), ("text", "bad_tint"), ("text", "retry_tint"),
    ("text_muted", "ok_tint"), ("text_muted", "weak_tint"), ("text_muted", "bad_tint"), ("text_muted", "retry_tint"),
    ("bad_text", "raised"), ("bad_text", "bad_tint"), ("bad_text", "panel"),
] + [(f"{r}_fg", f"{r}_bg") for r in ROLES]


def _lum(hex_: str) -> float:
    h = hex_.lstrip("#")
    out = []
    for i in (0, 2, 4):
        c = int(h[i:i + 2], 16) / 255
        out.append(c / 12.92 if c <= 0.03928 else ((c + 0.055) / 1.055) ** 2.4)
    return 0.2126 * out[0] + 0.7152 * out[1] + 0.0722 * out[2]


def contrast(fg: str, bg: str) -> float:
    a, b = sorted((_lum(fg), _lum(bg)), reverse=True)
    return (a + 0.05) / (b + 0.05)


def contrast_report() -> list[tuple[str, str, float]]:
    return [(f, b, round(contrast(_P[f], _P[b]), 2)) for f, b in PAIRS]


def build_theme() -> gr.themes.Base:
    t = gr.themes.Base(primary_hue=gr.themes.colors.sky, neutral_hue=gr.themes.colors.slate,
                       font=[gr.themes.GoogleFont("Inter"), "ui-sans-serif", "system-ui", "sans-serif"],
                       font_mono=[gr.themes.GoogleFont("JetBrains Mono"), "ui-monospace", "monospace"],
                       text_size=gr.themes.sizes.text_md, radius_size=gr.themes.sizes.radius_md)
    v = {
        "body_background_fill": _P["page"], "body_text_color": _P["text"], "body_text_color_subdued": _P["text_muted"],
        "background_fill_primary": _P["panel"], "background_fill_secondary": _P["raised"],
        "block_background_fill": _P["panel"], "block_border_color": _P["border"], "block_label_text_color": _P["text_muted"],
        "block_label_background_fill": _P["raised"], "block_title_text_color": _P["text"], "block_info_text_color": _P["text_subtle"],
        "panel_background_fill": _P["panel"], "panel_border_color": _P["border"], "border_color_primary": _P["border"],
        "border_color_accent": _P["accent"], "color_accent_soft": _P["raised"], "code_background_fill": _P["code"],
        "input_background_fill": _P["raised"], "input_background_fill_focus": _P["raised"], "input_border_color": _P["border"],
        "input_border_color_focus": _P["accent"], "input_placeholder_color": _P["text_subtle"],
        "checkbox_label_background_fill": _P["raised"], "checkbox_label_background_fill_selected": _P["raised"],
        "checkbox_label_text_color": _P["text"], "checkbox_label_text_color_selected": _P["text"],
        "checkbox_label_border_color": _P["border"], "checkbox_label_border_color_selected": _P["accent"],
        "checkbox_background_color": _P["code"], "checkbox_background_color_selected": _P["accent"],
        "checkbox_border_color": _P["text_subtle"], "checkbox_border_color_selected": _P["accent"],
        "button_primary_background_fill": _P["accent"], "button_primary_background_fill_hover": "#a5d6ff",
        "button_primary_text_color": _P["on_accent"], "button_primary_text_color_hover": _P["on_accent"],
        "button_primary_border_color": _P["accent"],
        "button_secondary_background_fill": _P["raised"], "button_secondary_background_fill_hover": _P["border"],
        "button_secondary_text_color": _P["text"], "button_secondary_border_color": _P["border"],
        "slider_color": _P["accent"], "loader_color": _P["accent"], "link_text_color": _P["accent"],
        "link_text_color_hover": "#a5d6ff", "link_text_color_visited": _P["accent"],
        "table_text_color": _P["text"], "table_border_color": _P["border"], "table_even_background_fill": _P["panel"],
        "table_odd_background_fill": _P["raised"], "table_row_focus": _P["border"], "stat_background_fill": _P["raised"],
        "error_background_fill": _P["bad_tint"], "error_text_color": _P["bad_text"], "error_border_color": _P["bad_line"],
        "accordion_text_color": _P["text"], "shadow_spread": "0px", "block_shadow": "none",
    }
    t.set(**v, **{f"{k}_dark": val for k, val in v.items()})
    # anything not set explicitly: use the theme's dark value in light mode too, so both schemes render identically
    for k in list(vars(t)):
        if k.endswith("_dark") and k[:-5] not in v and getattr(t, k) is not None:  # None = "same as light"
            setattr(t, k[:-5], getattr(t, k))
    return t


def css() -> str:
    P = _P
    roles = "".join(f".ag-{r}{{background:{P[r + '_bg']};color:{P[r + '_fg']}}}" for r in ROLES)
    return f"""
:root,.dark{{color-scheme:dark}}
.gradio-container{{max-width:1480px!important;margin:0 auto!important;background:{P['page']}!important;color:{P['text']}}}
#pg-title h1{{font-size:28px;line-height:1.2;font-weight:800;margin:4px 0 2px;color:{P['text']}}}
#pg-title p{{font-size:15px;line-height:1.5;color:{P['text_muted']};margin:0}}
#pg-title strong{{color:{P['text']}}}
.pg-h{{font-size:16px;font-weight:700;color:{P['text']};margin:6px 0}}
.pg-banner{{border-radius:10px;padding:12px 16px;margin:0 0 4px}}
.pg-banner .head{{font-size:18px;font-weight:800;letter-spacing:.01em}}
.pg-banner .sub,.pg-banner .ds{{font-size:14px;line-height:1.45;margin-top:4px}}
.pg-banner code{{font-family:var(--font-mono);font-size:13px;background:rgba(0,0,0,.14);padding:0 4px;border-radius:4px;color:inherit}}
.pg-banner.rec{{background:{P['rec_bg']}}}.pg-banner.rec,.pg-banner.rec *{{color:{P['rec_fg']}!important}}
.pg-banner.live{{background:{P['live_bg']}}}.pg-banner.live,.pg-banner.live *{{color:{P['live_fg']}!important}}
.pg-banner.mixed{{background:{P['mixed_bg']}}}.pg-banner.mixed,.pg-banner.mixed *{{color:{P['mixed_fg']}!important}}
.pg-banner.idle{{background:{P['idle_bg']}}}.pg-banner.idle,.pg-banner.idle *{{color:{P['idle_fg']}!important}}
.pg-banner.rec .warnline,.pg-banner.rec .warnline *{{color:{P['warnline_fg']}!important;font-weight:700}}
.pg-banner .warnline{{font-weight:700}}
.tl{{max-height:680px;overflow:auto;border:1px solid {P['border']};border-radius:10px;background:{P['panel']}}}
.tl table{{width:100%;border-collapse:collapse;table-layout:fixed;font-size:14px;color:{P['text']};margin:0}}
.tl td{{padding:7px 8px;border:0;border-bottom:1px solid {P['border']};vertical-align:top;background:transparent;color:{P['text']}}}
.tl td.t{{color:{P['text_subtle']};white-space:nowrap;width:52px;font-size:12px;font-variant-numeric:tabular-nums}}
.tl td:nth-child(2){{width:106px}}
.tl .act{{font-weight:700;font-size:14px;color:{P['text']}}}
.tl .sum{{color:{P['text_muted']};word-break:break-word;font-size:13px;line-height:1.45;margin-top:2px}}
.tl tr.bad td{{background:{P['bad_tint']}}}.tl tr.bad td:first-child{{box-shadow:inset 4px 0 {P['bad_line']}}}
.tl tr.warn td{{background:{P['weak_tint']}}}.tl tr.warn td:first-child{{box-shadow:inset 4px 0 {P['weak_line']}}}
.tl tr.retry td{{background:{P['retry_tint']}}}.tl tr.retry td:first-child{{box-shadow:inset 4px 0 {P['retry_line']}}}
.tl tr.dim .act,.tl tr.dim .sum{{color:{P['text_subtle']};font-weight:500}}
.pill{{display:inline-block;border-radius:999px;padding:1px 9px;font-size:12px;font-weight:800;line-height:18px;letter-spacing:.02em;white-space:nowrap;vertical-align:1px}}
.pill *{{color:inherit!important}}
.pill.ok{{background:{P['ok_bg']};color:{P['ok_fg']}}}.pill.bad{{background:{P['bad_bg']};color:{P['bad_fg']}}}
.pill.warn{{background:{P['weak_bg']};color:{P['weak_fg']}}}.pill.retry{{background:{P['retry_bg']};color:{P['retry_fg']}}}
.pill.muted{{background:{P['muted_bg']};color:{P['muted_fg']}}}
.ag{{display:inline-block;font-weight:800;font-size:12px;border-radius:6px;padding:2px 7px;line-height:16px}}
{roles}
.pg-panel{{color:{P['text']};font-size:14px;line-height:1.5}}
.pg-panel h4{{font-size:15px;font-weight:700;color:{P['text']};margin:10px 0 6px}}
.pg-panel li{{margin:6px 0;color:{P['text']}}}
.pg-panel code,.step code,.tl code{{font-family:var(--font-mono);font-size:12.5px;color:{P['accent']};background:{P['code']};padding:1px 5px;border-radius:4px}}
.step{{border:1px solid {P['border']};border-radius:10px;padding:10px 12px;margin-bottom:10px;background:{P['panel']};color:{P['text']}}}
.step .sh{{font-size:14px;color:{P['text']}}}
.step pre{{white-space:pre-wrap;font-family:var(--font-mono);font-size:12.5px;line-height:1.5;background:{P['code']};color:{P['text']};padding:8px 10px;border-radius:8px;margin:4px 0;border:1px solid {P['border']}}}
.small{{font-size:13px;color:{P['text_muted']};line-height:1.45}}.badtxt{{color:{P['bad_text']}!important}}
.cv{{padding:8px 10px;border-left:5px solid {P['border']};margin:6px 0;font-size:14px;line-height:1.45;border-radius:6px;color:{P['text']};background:{P['raised']}}}
.cv.ok{{border-color:{P['ok_line']};background:{P['ok_tint']}}}.cv.bad{{border-color:{P['bad_line']};background:{P['bad_tint']}}}
.cv.warn{{border-color:{P['weak_line']};background:{P['weak_tint']}}}.cv.retry{{border-color:{P['retry_line']};background:{P['retry_tint']}}}
.cv.branch{{font-weight:700}}
.empty{{color:{P['text_muted']};padding:18px;font-size:14px;border:1px dashed {P['border']};border-radius:10px;background:{P['panel']};text-align:center}}
.hint{{color:{P['text_subtle']};font-size:12px;text-align:right;margin-top:4px}}
#answer-card{{background:{P['raised']};border:1px solid {P['border']};border-left:6px solid {P['accent']};border-radius:12px;padding:14px 20px!important;color:{P['text']}}}
#answer-card h3{{font-size:20px;line-height:1.3;font-weight:800;color:{P['text']};margin:2px 0 8px}}
#answer-card p,#answer-card li{{font-size:15px;line-height:1.55;color:{P['text']}}}
#answer-card strong{{color:{P['text']}}}
#answer-card code{{font-family:var(--font-mono);font-size:12.5px;color:{P['accent']};background:{P['code']};padding:1px 6px;border-radius:4px}}
#answer-card em{{color:{P['text_muted']}}}
#fixture-status p{{font-size:14px;color:{P['text_muted']}}}
#pg-footnote p{{font-size:13px;color:{P['text_subtle']}}}
#pg-footnote code{{color:{P['accent']};background:{P['code']}}}
@media (max-width: 900px){{.tl{{max-height:520px}}.tl td:nth-child(2){{width:84px}}#pg-title h1{{font-size:23px}}}}
"""


# Forces the same rendering whatever the browser's colour scheme (also covers Gradio's ?__theme=light).
FORCE_DARK_JS = """
() => {
  const apply = () => { document.documentElement.classList.add('dark'); document.body && document.body.classList.add('dark'); };
  apply();
  new MutationObserver(apply).observe(document.body, {attributes: true, attributeFilter: ['class']});
}
"""
