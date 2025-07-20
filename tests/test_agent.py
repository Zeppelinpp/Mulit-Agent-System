import asyncio
from openai import AsyncOpenAI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.agent import PlanningAgent, AgentWorker, AgentRunner

# ==== 定义模拟工具函数（实际业务中你可以替换成真实的工具） ====

def load_csv(file_path: str):
    print(f"Loading CSV from {file_path}")
    return {"columns": ["col1", "col2"], "rows": 100}

def clean_data(columns: list):
    print(f"Cleaning data for columns: {columns}")
    return {"cleaned": True}

def describe_data():
    print("Generating statistics")
    return {"mean": 10, "std": 2}

def plot_data():
    print("Plotting data")
    return "plot.png"

# ==== 初始化智能体 ====

client = AsyncOpenAI(
    api_key="sk-4cd77baf9cdb43aeadb180887e28ad22",  # 替换为你的 key
    base_url="https://api.deepseek.com",  # 适配你实际部署的 API
)

worker_1 = AgentWorker(
    name="DataAgent",
    description="Handles data ingestion and preprocessing",
    model="deepseek-chat",
    client=client,
    system_prompt="You are a data processing expert.",
    tools=[load_csv, clean_data],
)

worker_2 = AgentWorker(
    name="AnalysisAgent",
    description="Performs statistics and visualization",
    model="deepseek-chat",
    client=client,
    system_prompt="You are a data analysis expert.",
    tools=[describe_data, plot_data],
)

workers = {
    "DataAgent": worker_1,
    "AnalysisAgent": worker_2,
}

workers_spec = [worker_1.description, worker_2.description]

planner = PlanningAgent(name="Planner", client=client, workers=workers_spec)

runner = AgentRunner(planner=planner, workers=workers)

# ==== 启动任务 ====

async def main():
    query = "请完成一个简单的数据分析，包括读取CSV、数据清洗、描述统计和绘图。"
    final_state = await runner.run(query)

    print("\n\n=== 最终状态 ===")
    print("成功任务:", [t.task_name for t in final_state.success_tasks])
    print("失败任务:", [t.task_name for t in final_state.failed_tasks])

if __name__ == "__main__":
    asyncio.run(main())