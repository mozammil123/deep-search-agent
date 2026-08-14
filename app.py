"""
app.py

Gradio front-end for the research pipeline defined in research_agent.py.
Run with: python app.py
"""

import logging
import gradio as gr
from research_agent import research_pipeline

# Log full details server-side only (never returned to the browser).
# On Spaces this goes to the container's Logs tab, which only you can see.
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("app")

GENERIC_ERROR_MESSAGE = (
    "⚠️ Something went wrong while researching that query. "
    "Please try again in a moment."
)


async def run_research(query: str, progress=gr.Progress()):
    """
    Async generator bridging research_pipeline() to Gradio outputs.
    Yields (status_md, report_md, followups_md) so the UI updates live.

    Any exception is caught here so that raw SDK/HTTP error text — which
    can sometimes include request headers or other sensitive details —
    is never shown to the user or returned as a Gradio error. Only a
    generic message reaches the UI; full details go to server-side logs.
    """
    plan_lines = ""

    try:
        async for stage, fraction, message, payload in research_pipeline(query):
            progress(fraction, desc=message)

            if stage == "error":
                yield f"⚠️ {message}", "", ""
                return

            if stage == "planning":
                yield f"🧭 **{message}**", "", ""

            elif stage == "planned":
                plan = payload
                plan_lines = "\n".join(f"- **{s.query}** — {s.reason}" for s in plan.searches)
                yield f"🧭 **{message}**\n\n{plan_lines}", "", ""

            elif stage == "searching":
                yield f"🔎 **{message}**\n\n{plan_lines}", "", ""

            elif stage == "writing":
                yield f"✍️ **{message}**", "", ""

            elif stage == "done":
                report = payload
                followups_md = "\n".join(f"- {q}" for q in report.follow_up_questions)
                yield (
                    f"✅ **Done.**\n\n**Summary:** {report.short_summary}",
                    report.markdown_report,
                    followups_md,
                )

    except Exception:
        # exc_info=True logs the full traceback server-side for debugging,
        # without ever exposing it to the client.
        logger.exception("research_pipeline failed for query: %r", query)
        yield GENERIC_ERROR_MESSAGE, "", ""


def save_report(report_md: str):
    if not report_md or not report_md.strip():
        return gr.update(visible=False)
    path = "report.md"
    with open(path, "w", encoding="utf-8") as f:
        f.write(report_md)
    return gr.update(value=path, visible=True)


def clear_all():
    return "", "", "", "", gr.update(visible=False)


with gr.Blocks(title="AI Research Agent", theme=gr.themes.Soft()) as demo:
    gr.Markdown(
        """
        # 🔬 AI Research Agent
        Ask a research question. The agent will plan searches, search the web,
        and write a detailed markdown report for you.
        """
    )

    with gr.Row():
        query_box = gr.Textbox(
            label="Research question",
            placeholder="e.g. Most popular AI Agent frameworks in 2026",
            scale=4,
        )
        submit_btn = gr.Button("Research", variant="primary", scale=1)

    with gr.Row():
        clear_btn = gr.Button("Clear")

    status_box = gr.Markdown(label="Status")

    with gr.Tab("Report"):
        report_box = gr.Markdown()

    with gr.Tab("Follow-up questions"):
        followups_box = gr.Markdown()

    download_file = gr.File(label="Download report (.md)", visible=False)

    submit_event = submit_btn.click(
        fn=run_research,
        inputs=query_box,
        outputs=[status_box, report_box, followups_box],
    ).then(
        fn=save_report,
        inputs=report_box,
        outputs=download_file,
    )

    query_box.submit(
        fn=run_research,
        inputs=query_box,
        outputs=[status_box, report_box, followups_box],
    ).then(
        fn=save_report,
        inputs=report_box,
        outputs=download_file,
    )

    clear_btn.click(
        fn=clear_all,
        outputs=[query_box, status_box, report_box, followups_box, download_file],
    )

if __name__ == "__main__":
    # show_error=False stops raw exception text (which can include SDK/HTTP
    # error details) from being displayed to users in the browser. We handle
    # and sanitize all errors ourselves in run_research() above.
    demo.queue().launch(show_error=False)
