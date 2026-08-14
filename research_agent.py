"""
research_agent.py

Core orchestration logic for the AI research pipeline:
Planner Agent -> Search Agent(s) -> Writer Agent.

This module has no UI code in it. Import `plan_searches`, `run_search`,
and `write_report` (or the convenience `research_pipeline` generator)
from a UI layer such as app.py.
"""

from agents import Agent, Runner, function_tool, OpenAIChatCompletionsModel, set_tracing_disabled
from agents.model_settings import ModelSettings
from openai import AsyncOpenAI
from dotenv import load_dotenv
from ddgs import DDGS
from pydantic import BaseModel, Field
import os
import asyncio
import json

load_dotenv()

# Fail fast with a clear, non-sensitive message if secrets are missing,
# instead of letting a raw KeyError (or a downstream SDK error that may
# echo request details) surface later.
_GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY")
_MODEL = os.environ.get("MODEL")

if not _GEMINI_API_KEY:
    raise RuntimeError(
        "GEMINI_API_KEY is not set. Set it as an environment variable "
        "(locally via .env, or as a Space secret in production) — never "
        "hardcode it in source."
    )
if not _MODEL:
    raise RuntimeError(
        "MODEL is not set. Set it as an environment variable "
        "(locally via .env, or as a Space secret in production)."
    )

gemini_client = AsyncOpenAI(
    api_key=_GEMINI_API_KEY,
    base_url="https://generativelanguage.googleapis.com/v1beta/openai/",
)

set_tracing_disabled(True)

gemini_model = OpenAIChatCompletionsModel(
    model=_MODEL,
    openai_client=gemini_client,
)

search_settings = ModelSettings(tool_choice="required")


# ---------- Tool ----------
@function_tool
def web_search_tool(input: str) -> str:
    """Search the web for the input provided."""
    results = DDGS().text(input, max_results=3)
    return json.dumps(results)


# ---------- Schemas ----------
class WebSearchItem(BaseModel):
    reason: str = Field(description="Why this search matters for the query.")
    query: str = Field(description="The search query to run.")


class WebSearchPlan(BaseModel):
    searches: list[WebSearchItem] = Field(description="A list of web searches to perform.")


class ReportData(BaseModel):
    short_summary: str = Field(description="A 2-3 sentence summary of the findings.")
    markdown_report: str = Field(description="The final markdown report.")
    follow_up_questions: list[str] = Field(description="Suggested follow-up research topics.")


# ---------- Agents ----------
planner_agent = Agent(
    name="Planner Agent",
    instructions=(
        "You are a research assistant. Given a user query, come up with a set of web "
        "searches to perform to best answer the query. Output 5 search terms."
    ),
    model=gemini_model,
    output_type=WebSearchPlan,
)

search_agent = Agent(
    name="Search Agent",
    instructions=(
        "You are an agent who searches the web to answer a query, then summarizes the "
        "result in 2-3 concise paragraphs capturing the key facts."
    ),
    tools=[web_search_tool],
    model=gemini_model,
    model_settings=search_settings,
)

writer_agent = Agent(
    name="Writer Agent",
    instructions=(
        "You are a senior researcher writing a cohesive report for a research query. "
        "You'll be given the original query and a set of research summaries. Write a "
        "comprehensive, well-structured markdown report of at least 1000 words, "
        "with headings, based on the research."
    ),
    model=gemini_model,
    output_type=ReportData,
)


# ---------- Orchestration ----------
async def plan_searches(query: str) -> WebSearchPlan:
    result = await Runner.run(planner_agent, query)
    return result.final_output


async def run_search(item: WebSearchItem) -> str:
    prompt = f"Search term: {item.query}\nReason for search: {item.reason}"
    result = await Runner.run(search_agent, prompt)
    return result.final_output


async def write_report(query: str, search_results: list[str]) -> ReportData:
    prompt = (
        f"Original query: {query}\n\n"
        "Research summaries:\n"
        + "\n\n".join(f"[{i+1}] {r}" for i, r in enumerate(search_results))
    )
    result = await Runner.run(writer_agent, prompt)
    return result.final_output


async def research_pipeline(query: str):
    """
    Runs the full plan -> search -> write pipeline and yields progress
    updates as (stage, fraction_complete, message, payload) tuples so a UI
    layer can render live status. `payload` is None until the relevant
    stage produces data (plan, a single search result, or the final report).
    """
    if not query or not query.strip():
        yield ("error", 0.0, "Please enter a research question.", None)
        return

    # Stage 1: Planning
    yield ("planning", 0.05, "Planning searches...", None)
    plan = await plan_searches(query)
    yield ("planned", 0.2, f"Planned {len(plan.searches)} searches.", plan)

    # Stage 2: Searching (concurrent, incremental progress)
    total = len(plan.searches)
    completed = 0
    search_results = []

    tasks = [asyncio.create_task(run_search(item)) for item in plan.searches]
    for coro in asyncio.as_completed(tasks):
        result = await coro
        search_results.append(result)
        completed += 1
        fraction = 0.2 + 0.5 * (completed / total)
        yield ("searching", fraction, f"Searching the web... ({completed}/{total})", None)

    # Stage 3: Writing
    yield ("writing", 0.85, "Writing the final report...", None)
    report = await write_report(query, search_results)

    yield ("done", 1.0, "Done.", report)


async def run_research_sync(query: str) -> ReportData:
    """Convenience one-shot version (no progress streaming) for CLI/testing use."""
    plan = await plan_searches(query)
    search_results = await asyncio.gather(*(run_search(item) for item in plan.searches))
    return await write_report(query, search_results)


if __name__ == "__main__":
    # Simple CLI smoke test: python research_agent.py
    q = "Most popular AI Agent frameworks in 2026"
    report = asyncio.run(run_research_sync(q))
    print(report.short_summary)
    with open("report.md", "w", encoding="utf-8") as f:
        f.write(report.markdown_report)
    print("Saved to report.md")
