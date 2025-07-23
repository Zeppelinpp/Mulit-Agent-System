import asyncio
from openai import AsyncOpenAI
from dotenv import load_dotenv

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.agent import PlanningAgent, AgentWorker, AgentRunner
from core.tools import web_search

load_dotenv()

# ==== 定义模拟工具函数（实际业务中你可以替换成真实的工具） ====

def load_csv(file_path: str):
    return {"columns": ["col1", "col2"], "rows": 100}

def clean_data(columns: list):
    return {"cleaned": True}

def describe_data():
    return {"mean": 10, "std": 2}

def plot_data():
    return "plot.png"

# ==== 初始化智能体 ====

client = AsyncOpenAI(
    api_key=os.getenv("QWEN_API_KEY"),
    base_url=os.getenv("QWEN_BASE_URL"),
)

worker_1 = AgentWorker(
    name="Researcher_1",
    description="Comprehensive research on certain topic",
    model="qwen-plus-latest",
    client=client,
    system_prompt="You are a data processing expert.",
    tools=[web_search],
)

worker_2 = AgentWorker(
    name="Researcher_2",
    description="Comprehensive research on certain topic",
    model="qwen-plus-latest",
    client=client,
    system_prompt="You are a data analysis expert.",
    tools=[web_search],
)

workers = {
    "Researcher_1": worker_1,
    "Researcher_2": worker_2,
}

workers_spec = [worker_1.description, worker_2.description]

planner = PlanningAgent(name="Planner", client=client, model="qwen-plus-latest", workers=workers_spec)

runner = AgentRunner(planner=planner, workers=workers)

# ==== 启动任务 ====

async def main():
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    
    console = Console()
    
    query = "How to build a multi-agent system using asyncio queue? And what's behind the hodd of asyncio queue."
    final_state = await runner.run(query)

    # Create elegant final results display
    results_table = Table(show_header=True, header_style="bold magenta", border_style="blue", show_lines=True)
    results_table.add_column("Task Status", style="bold", width=15)
    results_table.add_column("Task Name", style="cyan", width=35)
    
    if final_state.success_tasks:
        for task in final_state.success_tasks:
            results_table.add_row("✅ Completed", task.task_name)
    
    if final_state.failed_tasks:
        for task in final_state.failed_tasks:
            results_table.add_row("❌ Failed", task.task_name)
    
    # Display final results
    console.print("\n")
    console.print(Panel(
        results_table,
        title="[bold blue]🎯 Workflow Execution Results[/bold blue]",
        border_style="blue",
        padding=(1, 2)
    ))
    
    # Summary message
    total_tasks = len(final_state.success_tasks) + len(final_state.failed_tasks)
    success_rate = len(final_state.success_tasks) / total_tasks * 100 if total_tasks > 0 else 0
    
    if success_rate == 100:
        summary_style = "bold green"
        summary_icon = "🎉"
        summary_text = f"All {total_tasks} tasks completed successfully!"
    elif success_rate >= 50:
        summary_style = "bold yellow"
        summary_icon = "⚠️"
        summary_text = f"{len(final_state.success_tasks)}/{total_tasks} tasks completed successfully ({success_rate:.0f}% success rate)"
    else:
        summary_style = "bold red"
        summary_icon = "🚫"
        summary_text = f"{len(final_state.success_tasks)}/{total_tasks} tasks completed successfully ({success_rate:.0f}% success rate)"
    
    console.print(Panel(
        f"[{summary_style}]{summary_icon} {summary_text}[/{summary_style}]",
        border_style=summary_style.split()[-1],
        padding=(0, 1)
    ))

if __name__ == "__main__":
    asyncio.run(main())