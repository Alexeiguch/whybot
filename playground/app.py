"""WhyBot playground UI (Gradio). Run: python -m playground.app   (from the repo root)

Replay (default): recorded real runs through the WhyBot HTTP API, no keys. Live: your NVIDIA key, in memory only.
"""
from __future__ import annotations

import os
import sys
import time
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import gradio as gr  # noqa: E402

from playground import core  # noqa: E402

CUSTOM = "✎ Type my own question below"
EV_HEADERS = ["qid", "hypothesis", "status", "source table(s)", "sql_hash", "rows", "provenance", "recorded_at", "preview"]

CSS = """
.pg-banner{border-radius:10px;padding:10px 14px;margin:2px 0 6px 0;border:2px solid}
.pg-banner .head{font-weight:800;font-size:1.05em;letter-spacing:.02em}
.pg-banner .sub,.pg-banner .ds{font-size:.88em;margin-top:2px}
.pg-banner.rec{background:#fff8db;border-color:#e0b100;color:#4a3b00}
.pg-banner.live{background:#e5f8ea;border-color:#22a447;color:#0d3d1b}
.pg-banner.mixed{background:#e8f1ff;border-color:#3b7be0;color:#0f2a55}
.pg-banner.idle{background:#f3f4f6;border-color:#c9ccd1;color:#333}
.warnline{color:#a35200}
.tl{max-height:640px;overflow:auto;border:1px solid #ddd;border-radius:8px}
.tl table{width:100%;border-collapse:collapse;font-size:12.5px}
.tl td{padding:3px 6px;border-bottom:1px solid #eee;vertical-align:top}
.tl table{table-layout:fixed}.tl td.t{color:#888;white-space:nowrap;width:46px;font-size:11px}.tl td:nth-child(2){width:92px}
.tl .act{font-family:monospace;font-weight:600}.tl .sum{color:#333;word-break:break-word;font-size:12px}
.tl tr.bad{background:#fdecec}.tl tr.warn{background:#fff4e0}.tl tr.dim{opacity:.55}
.pill{display:inline-block;border-radius:9px;padding:0 7px;font-size:11px;font-weight:700;color:#fff;background:#888}
.pill.ok{background:#22a447}.pill.bad{background:#d93025}.pill.warn{background:#e8890c}.pill.muted{background:#8a8f98}
.ag{font-weight:700;font-size:11.5px;border-radius:5px;padding:0 5px;background:#eef}
.ag-Planner{background:#e9e3ff}.ag-Executor{background:#dff1ff}.ag-Analyst{background:#e6f7ea}.ag-Critic{background:#ffe5e0}
.ag-Writer{background:#fff1cc}.ag-Memory{background:#eeeeee}.ag-Orchestrator{background:#f2f2f2}
.step{border:1px solid #ddd;border-radius:8px;padding:6px 8px;margin-bottom:8px}
.step pre{white-space:pre-wrap;font-size:12px;background:#f7f7f9;padding:6px;border-radius:6px;margin:2px 0}
.small{font-size:11.5px;color:#555}.badtxt{color:#c62828}
.cv{padding:5px 8px;border-left:4px solid #ccc;margin:4px 0;font-size:13px;border-radius:4px}
.cv.ok{border-color:#22a447;background:#f1fbf3}.cv.bad{border-color:#d93025;background:#fdecec;font-weight:600}
.cv.warn{border-color:#e8890c;background:#fff4e0}.cv.branch{font-weight:700}
.empty{color:#888;padding:14px;font-style:italic}.hint{color:#999;font-size:11px;text-align:right}
"""


def _choices() -> list[str]:
    rec = core.replay_status()["recorded"]
    out = [f"{q}  [recorded]" if q in rec else q for q in core.sample_questions()]
    return out + [CUSTOM]


def _strip(choice: str) -> str:
    return choice.replace("  [recorded]", "").strip()


def fixture_md() -> str:
    st = core.replay_status()
    zet_ok, reason = core.zetaris_configured()
    if st["ready"]:
        fx = f"✅ Recorded runs found: {len(st['recorded'])} question(s), {st['queries']} recorded Zetaris results."
    elif st["catalog"]:
        fx = (f"⏳ {st['queries']} recorded Zetaris results found, but the recorded Nemotron responses are not committed yet. "
              "They are being re-recorded; this panel updates by itself once they land.")
    else:
        fx = "⏳ Recorded runs are not in this checkout yet (being re-recorded). They are picked up automatically once committed."
    return f"{fx}\n\nZetaris for Live: {'✅' if zet_ok else '➖'} {reason}."


def idle_banner() -> str:
    return ("<div class='pg-banner idle'><div class='head'>Ready</div><div class='sub'>Replay shows a <b>RECORDED</b> real run "
            "(no keys). Live calls NVIDIA Nemotron with your key.</div></div>")


def investigate(choice, custom_q, mode, api_key, pacing):
    question = (custom_q or "").strip() if choice == CUSTOM else _strip(choice or "")
    mode_key = "replay" if mode.startswith("Replay") else "live"
    view = core.RunView(mode=mode_key, question=question, run_id=core.new_run_id(mode_key))

    def frame():
        return (view.banner() if view.events else idle_banner(), view.timeline(), view.hypotheses(), view.sql_steps(),
                view.critic(), gr.update(value=view.evidence_rows() or None), view.answer(), view.write_trace())

    if len(question) < 5:
        view.notice = "**Pick a sample question, or type a question of at least 5 characters.**"
        yield frame()
        return
    if mode_key == "replay":
        st = core.replay_status(question)
        if not st["question_ready"]:
            view.notice = core.friendly_missing(question, st)
            yield frame()
            return
    try:
        stream = core.stream_replay(question, view.run_id) if mode_key == "replay" else core.stream_live(question, view.run_id, api_key)
        api_key = None  # nothing below needs it; the live worker holds its own reference for the run only
        last = 0.0
        for kind, payload in stream:
            if kind == "trace":
                view.add(payload)
                if mode_key == "replay" and pacing:
                    time.sleep(float(pacing))
                if time.time() - last > 0.25 or mode_key == "replay":
                    last = time.time()
                    yield frame()
            elif kind == "result":
                view.result = payload
        yield frame()
    except Exception as e:  # never show a stack trace, never echo inputs
        msg = str(e)
        if api_key:
            msg = msg.replace(api_key.strip(), "<redacted>")
        view.notice = f"**Could not run:** {core._e(msg[:400])}"
        yield frame()


def build() -> gr.Blocks:
    with gr.Blocks(title="WhyBot Playground") as demo:
        gr.Markdown("# WhyBot Playground\n**Why did the number move?** A Planner, Executor, Analyst, Critic and Writer investigate "
                    "with SQL federated through **Zetaris**, memory from **Meterless**, and reasoning by **NVIDIA Nemotron**. "
                    "Every claim cites its query id, SQL hash and source table.")
        with gr.Row():
            with gr.Column(scale=1, min_width=320):
                choices = _choices()
                q = gr.Dropdown(choices, value=choices[0], label="Question", allow_custom_value=False)
                custom = gr.Textbox(label="Your own question", placeholder="Why did NYC yellow taxi trips drop on …?",
                                    lines=2, visible=False)
                mode = gr.Radio(["Replay (recorded real run, no keys)", "Live (your NVIDIA key)"],
                                value="Replay (recorded real run, no keys)", label="Mode")
                key = gr.Textbox(label="NVIDIA API key (nvapi-…)", type="password", visible=False,
                                 info="Used for this run only, kept in memory, never logged or stored.")
                pacing = gr.Slider(0, 0.5, value=0.08, step=0.02, label="Replay pacing (s per event)",
                                   info="Replays finish in seconds; slow them down to watch the agents.")
                run = gr.Button("Investigate", variant="primary")
                status = gr.Markdown(fixture_md())
                refresh = gr.Button("Re-check recorded runs", size="sm")
                trace = gr.File(label="Download the JSONL trace", interactive=False)
                gr.Markdown("API: `POST /run`, `POST /runs` + SSE, see `docs/api.md`. Built-in UI on port 8000.")
            with gr.Column(scale=3):
                banner = gr.HTML(idle_banner())
                answer = gr.Markdown()
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("#### Agent timeline")
                        timeline = gr.HTML(core.RunView("replay", "", "").timeline())
                    with gr.Column(scale=6):
                        with gr.Tabs():
                            with gr.Tab("Critic"):
                                critic = gr.HTML()
                            with gr.Tab("Hypotheses"):
                                hyps = gr.HTML()
                            with gr.Tab("SQL steps"):
                                sql = gr.HTML()
                            with gr.Tab("Evidence"):
                                evidence = gr.Dataframe(headers=EV_HEADERS, wrap=True, interactive=False)

        def on_choice(c):
            return gr.update(visible=c == CUSTOM)

        def on_mode(m):
            return gr.update(visible=m.startswith("Live"))

        def on_refresh():
            ch = _choices()
            return fixture_md(), gr.update(choices=ch)

        q.change(on_choice, q, custom)
        mode.change(on_mode, mode, key)
        refresh.click(on_refresh, None, [status, q])
        run.click(investigate, [q, custom, mode, key, pacing],
                  [banner, timeline, hyps, sql, critic, evidence, answer, trace], concurrency_limit=4)
        run.click(fixture_md, None, status)
        demo.load(on_refresh, None, [status, q])
    return demo


def main():
    os.environ.setdefault("SAMPLE_MODE", "true")
    try:
        url = core.ensure_api()
        print(f"[playground] WhyBot API at {url}", flush=True)
    except Exception as e:
        print(f"[playground] WARN: WhyBot API not available yet: {e}", flush=True)
    demo = build()
    demo.queue(default_concurrency_limit=4)
    demo.launch(server_name=os.environ.get("GRADIO_SERVER_NAME", "127.0.0.1"),
                server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
                allowed_paths=[str(core.TRACE_DIR)], show_error=False, css=CSS)


if __name__ == "__main__":
    main()
