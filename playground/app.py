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

from playground import core, theme  # noqa: E402

CUSTOM = "✎ Type my own question below"
EV_HEADERS = ["qid", "hypothesis", "status", "source table(s)", "sql_hash", "rows", "provenance", "recorded_at", "preview"]



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
    return ("<div class='pg-banner idle' role='status'><div class='head'>● Ready</div><div class='sub'>Replay shows a <b>RECORDED</b> real run "
            "(no keys). Live calls NVIDIA Nemotron with your key.</div></div>")


ANSWER_EMPTY = ("### The cited answer appears here\n\n_Pick a question and press **Investigate**. When the Writer finishes, "
                "the headline, recommendation and every claim with its `[query id · SQL hash · table]` citation show up in this card._")


def investigate(choice, custom_q, mode, api_key, pacing):
    question = (custom_q or "").strip() if choice == CUSTOM else _strip(choice or "")
    mode_key = "replay" if mode.startswith("Replay") else "live"
    view = core.RunView(mode=mode_key, question=question, run_id=core.new_run_id(mode_key))

    def frame():
        return (view.banner() if view.events else idle_banner(), view.timeline(), view.hypotheses(), view.sql_steps(),
                view.critic(), gr.update(value=view.evidence_rows() or None), view.answer() or ANSWER_EMPTY, view.write_trace())

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


THEME = theme.build_theme()
_EMPTY = core.RunView("replay", "", "")


def build() -> gr.Blocks:
    with gr.Blocks(title="WhyBot Playground") as demo:
        gr.Markdown("# WhyBot Playground\n**Why did the number move?** A Planner, Executor, Analyst, Critic and Writer investigate "
                    "with SQL federated through **Zetaris**, memory from **Meterless**, and reasoning by **NVIDIA Nemotron**. "
                    "Every claim cites its query id, SQL hash and source table.", elem_id="pg-title")
        with gr.Row(equal_height=False):
            with gr.Column(scale=1, min_width=300):
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
                status = gr.Markdown(fixture_md(), elem_id="fixture-status")
                refresh = gr.Button("Re-check recorded runs", size="sm")
                trace = gr.File(label="Download the JSONL trace", interactive=False, height=110)
                gr.Markdown("API: `POST /run`, `POST /runs` + SSE, see `docs/api.md`. Built-in UI on port 8000.", elem_id="pg-footnote")
            with gr.Column(scale=3):
                banner = gr.HTML(idle_banner())
                answer = gr.Markdown(ANSWER_EMPTY, elem_id="answer-card")
                with gr.Row():
                    with gr.Column(scale=5, min_width=340):
                        gr.HTML("<div class='pg-h'>Agent timeline</div>")
                        timeline = gr.HTML(_EMPTY.timeline())
                    with gr.Column(scale=6, min_width=340):
                        with gr.Tabs():
                            with gr.Tab("Critic"):
                                critic = gr.HTML(_EMPTY.critic())
                            with gr.Tab("Hypotheses"):
                                hyps = gr.HTML(_EMPTY.hypotheses())
                            with gr.Tab("SQL steps"):
                                sql = gr.HTML(_EMPTY.sql_steps())
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
                allowed_paths=[str(core.TRACE_DIR)], show_error=False,
                theme=THEME, css=theme.css(), js=theme.FORCE_DARK_JS)


if __name__ == "__main__":
    main()
