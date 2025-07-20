import pytest
import asyncio
import json
from unittest.mock import AsyncMock, MagicMock, patch
from typing import List, Callable
from openai import AsyncOpenAI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.agent import AgentWorker, Task
from core.utils import ToolConverter


# Mock tools for testing
def mock_calculator(operation: str, a: float, b: float) -> float:
    """A simple calculator tool for testing."""
    if operation == "add":
        return a + b
    elif operation == "subtract":
        return a - b
    elif operation == "multiply":
        return a * b
    elif operation == "divide":
        if b == 0:
            raise ValueError("Cannot divide by zero")
        return a / b
    else:
        raise ValueError(f"Unknown operation: {operation}")


def mock_weather_tool(city: str) -> str:
    """A mock weather tool for testing."""
    weather_data = {
        "New York": "Sunny, 25°C",
        "London": "Rainy, 15°C",
        "Tokyo": "Cloudy, 22°C"
    }
    return weather_data.get(city, "Weather data not available")


def mock_search_tool(query: str, limit: int = 5) -> List[str]:
    """A mock search tool for testing."""
    return [f"Result {i} for '{query}'" for i in range(1, limit + 1)]


class TestAgentWorker:
    """Test cases for AgentWorker class."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        client = AsyncMock(spec=AsyncOpenAI)
        # Mock the chat.completions.create method to return an awaitable
        client.chat.completions.create = AsyncMock()
        return client

    @pytest.fixture
    def sample_tools(self):
        """Sample tools for testing."""
        return [mock_calculator, mock_weather_tool, mock_search_tool]

    @pytest.fixture
    def agent_worker(self, mock_client, sample_tools):
        """Create an AgentWorker instance for testing."""
        system_prompt = "You are a helpful assistant that can use tools to solve tasks."
        return AgentWorker(
            name="test_worker",
            description="A test worker agent",
            model="deepseek-chat",
            client=mock_client,
            system_prompt=system_prompt,
            tools=sample_tools
        )

    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            id="task_001",
            task_name="Calculate sum",
            goal="Calculate the sum of 5 and 3",
            result="",
            agent_name="test_worker",
            context={"numbers": [5, 3]}
        )

    def test_agent_worker_initialization(self, agent_worker):
        """Test AgentWorker initialization."""
        assert agent_worker.name == "test_worker"
        assert agent_worker.description == "A test worker agent"
        assert agent_worker.model == "deepseek-chat"
        assert agent_worker.system_prompt == "You are a helpful assistant that can use tools to solve tasks."
        assert len(agent_worker.tools) == 3
        assert "mock_calculator" in agent_worker.tools_registry
        assert "mock_weather_tool" in agent_worker.tools_registry
        assert "mock_search_tool" in agent_worker.tools_registry

    def test_setup_tools(self, agent_worker):
        """Test tool setup and conversion."""
        tools = agent_worker._setup_tools()
        assert len(tools) == 3
        
        # Check calculator tool schema
        calculator_tool = next(t for t in tools if t["function"]["name"] == "mock_calculator")
        assert calculator_tool["type"] == "function"
        assert "operation" in calculator_tool["function"]["parameters"]["properties"]
        assert "a" in calculator_tool["function"]["parameters"]["properties"]
        assert "b" in calculator_tool["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_execute_tool(self, agent_worker):
        """Test tool execution."""
        # Test calculator tool
        result = await agent_worker._execute_tool("mock_calculator", {
            "operation": "add",
            "a": 5,
            "b": 3
        })
        assert result == 8

        # Test weather tool
        result = await agent_worker._execute_tool("mock_weather_tool", {
            "city": "New York"
        })
        assert result == "Sunny, 25°C"

        # Test search tool
        result = await agent_worker._execute_tool("mock_search_tool", {
            "query": "test",
            "limit": 3
        })
        assert len(result) == 3
        assert "test" in result[0]

    @pytest.mark.asyncio
    async def test_execute_tool_with_error(self, agent_worker):
        """Test tool execution with errors."""
        # Test division by zero
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await agent_worker._execute_tool("mock_calculator", {
                "operation": "divide",
                "a": 5,
                "b": 0
            })

        # Test unknown operation
        with pytest.raises(ValueError, match="Unknown operation"):
            await agent_worker._execute_tool("mock_calculator", {
                "operation": "power",
                "a": 5,
                "b": 2
            })

        # Test non-existent tool
        with pytest.raises(KeyError):
            await agent_worker._execute_tool("non_existent_tool", {})

    @pytest.mark.asyncio
    async def test_handle_tool_calls(self, agent_worker):
        """Test handling multiple tool calls."""
        tool_calls = [
            {
                "id": "call_1",
                "name": "mock_calculator",
                "arguments": json.dumps({
                    "operation": "add",
                    "a": 2,
                    "b": 3
                })
            },
            {
                "id": "call_2",
                "name": "mock_weather_tool",
                "arguments": json.dumps({
                    "city": "London"
                })
            }
        ]

        results = await agent_worker._handle_tool_calls(tool_calls)
        assert len(results) == 2
        
        # Check first result
        assert results[0]["tool_call_id"] == "call_1"
        assert results[0]["tool_name"] == "mock_calculator"
        assert results[0]["tool_result"] == 5

        # Check second result
        assert results[1]["tool_call_id"] == "call_2"
        assert results[1]["tool_name"] == "mock_weather_tool"
        assert results[1]["tool_result"] == "Rainy, 15°C"

    @pytest.mark.asyncio
    async def test_run_task_without_tools(self, agent_worker, sample_task, mock_client):
        """Test running a task that doesn't require tools."""
        # Mock the API response to return a simple text response
        mock_response = MagicMock()
        mock_response.choices = [MagicMock()]
        mock_response.choices[0].message.content = "The sum of 5 and 3 is 8"
        mock_response.choices[0].message.tool_calls = None
        mock_client.chat.completions.create.return_value = mock_response

        result = await agent_worker.run(sample_task)

        assert result.status == "success"
        assert result.result == "The sum of 5 and 3 is 8"
        assert sample_task.status == "success"

    @pytest.mark.asyncio
    async def test_run_task_with_tools(self, agent_worker, sample_task, mock_client):
        """Test running a task that requires tool usage."""
        # Mock the API responses
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = None
        mock_response1.choices[0].message.tool_calls = [
            {
                "id": "call_1",
                "name": "mock_calculator",
                "arguments": json.dumps({
                    "operation": "add",
                    "a": 5,
                    "b": 3
                })
            }
        ]

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = "The calculation result is 8"
        mock_response2.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]

        result = await agent_worker.run(sample_task)

        assert result.status == "success"
        assert result.result == "The calculation result is 8"
        assert sample_task.status == "success"

        # Verify API was called twice
        assert mock_client.chat.completions.create.call_count == 2

    @pytest.mark.asyncio
    async def test_run_task_with_multiple_tool_calls(self, agent_worker, sample_task, mock_client):
        """Test running a task that requires multiple tool calls."""
        # Mock the API responses for multiple tool calls
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = None
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.tool_calls = [
            {
                "id": "call_1",
                "name": "mock_calculator",
                "arguments": json.dumps({
                    "operation": "add",
                    "a": 5,
                    "b": 3
                })
            },
            {
                "id": "call_2",
                "name": "mock_weather_tool",
                "arguments": json.dumps({
                    "city": "New York"
                })
            }
        ]

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = "Task completed successfully"
        mock_response2.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.side_effect = [mock_response1, mock_response2]

        result = await agent_worker.run(sample_task)

        assert result.status == "success"
        assert result.result == "Task completed successfully"
        assert sample_task.status == "success"

    @pytest.mark.asyncio
    async def test_get_context(self, agent_worker):
        """Test getting agent context."""
        context = await agent_worker._get_context()
        assert isinstance(context, dict)
        # Currently returns empty dict as per TODO comment
        assert context == {}

    def test_state_property(self, agent_worker):
        """Test state property access and modification."""
        # Test initial state
        state_dict = agent_worker.state
        assert state_dict["assigned_tasks"] is None
        assert len(state_dict["success_tasks"]) == 0
        assert len(state_dict["failed_tasks"]) == 0
        assert state_dict["retries"] == 0

        # Test state modification
        new_state = {
            "assigned_tasks": None,
            "success_tasks": [],
            "failed_tasks": [],
            "retries": 5
        }
        agent_worker.state = new_state
        assert agent_worker.state["retries"] == 5

    @pytest.mark.asyncio
    async def test_run_task_error_handling(self, agent_worker, sample_task, mock_client):
        """Test error handling during task execution."""
        # Mock API to raise an exception
        mock_client.chat.completions.create.side_effect = Exception("API Error")

        with pytest.raises(Exception, match="API Error"):
            await agent_worker.run(sample_task)

        # Task should remain in progress state
        assert sample_task.status == "in_progress"

    @pytest.mark.asyncio
    async def test_tool_execution_with_invalid_args(self, agent_worker):
        """Test tool execution with invalid arguments."""
        # Test missing required arguments
        with pytest.raises(TypeError):
            await agent_worker._execute_tool("mock_calculator", {
                "operation": "add"
                # Missing 'a' and 'b' arguments
            })

        # Test wrong argument types
        with pytest.raises(TypeError):
            await agent_worker._execute_tool("mock_calculator", {
                "operation": "add",
                "a": "not_a_number",
                "b": 3
            })


if __name__ == "__main__":
    # Run the tests
    pytest.main([__file__, "-v"]) 