# Multi-Agent System

A from-scratch implementation of a multi-agent system using Python's `asyncio.Queue()` for task management. Features a planner-worker architecture with support for both parallel and sequential task execution.

## 🏗️ Architecture

**Planner-Worker Pattern:**
- **PlanningAgent**: Analyzes queries and creates structured task lists
- **AgentWorker**: Executes tasks with specialized tools
- **AgentRunner**: Orchestrates parallel/sequential task execution
- **Task Management**: Status tracking with pending/in_progress/success/failure states

## 📁 Structure

```
core/
├── agent.py              # Main agent classes and task management
├── agent_prompts.py      # System prompts for agents  
└── utils.py              # Tool converters and utilities
tests/                    # Test suite
```

## 🚀 Features

- **Async Task Management**: Built on `asyncio.Queue()` with proper concurrency
- **Dynamic Task Planning**: Automatic sequential/parallel task categorization
- **Tool Integration**: Convert Python functions to OpenAI tool format
- **State Management**: Persistent agent state and task status tracking
- **Rich Output**: Beautiful console visualization using Rich library

## 🎯 Usage

```python
from core.agent import PlanningAgent, AgentWorker, AgentRunner

# Setup planner and workers
planner = PlanningAgent(name="planner", workers=[worker_configs])
worker = AgentWorker(name="worker", tools=[your_tools])
runner = AgentRunner(planner=planner, workers={"worker": worker})

# Execute
await runner.run("Your query here")
```

## 📚 Learning Journey

Personal project exploring:
- **Async Programming**: `asyncio` and concurrent task management
- **Multi-Agent Architectures**: Planner-worker patterns from scratch
- **LLM Integration**: OpenAI APIs and tool calling
- **System Design**: Scalable agent systems without frameworks

## 🔮 Future Goals

- **Network Mode**: Distributed agent communication
- **Advanced State Management**: Enhanced persistence and recovery
- **Memory Systems**: Long-term agent memory
- **Multi-modal Support**: Images, audio, and other data types

---

**Note**: Built from scratch without frameworks like LangGraph. Focus on understanding core concepts.
