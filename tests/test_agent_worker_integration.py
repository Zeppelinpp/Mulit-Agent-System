import pytest
import asyncio
from unittest.mock import AsyncMock, MagicMock
from openai import AsyncOpenAI

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from core.agent import AgentWorker, Task


# Real tools for integration testing
def calculator(operation: str, a: float, b: float) -> float:
    """A real calculator tool for integration testing."""
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


def weather_service(city: str) -> str:
    """A mock weather service for integration testing."""
    weather_data = {
        "New York": "Sunny, 25°C",
        "London": "Rainy, 15°C",
        "Tokyo": "Cloudy, 22°C",
        "Paris": "Partly cloudy, 20°C"
    }
    return weather_data.get(city, "Weather data not available")


class TestAgentWorkerIntegration:
    """Integration tests for AgentWorker with real tools."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client for integration testing."""
        client = AsyncMock(spec=AsyncOpenAI)
        client.chat.completions.create = AsyncMock()
        return client

    @pytest.fixture
    def agent_worker(self, mock_client):
        """Create an AgentWorker instance with real tools."""
        system_prompt = """
        You are a helpful assistant that can use tools to solve tasks.
        When given a task, analyze it and use the appropriate tools to complete it.
        Always provide clear and accurate results.
        """
        return AgentWorker(
            name="integration_worker",
            description="An integration test worker agent",
            model="deepseek-chat",
            client=mock_client,
            system_prompt=system_prompt,
            tools=[calculator, weather_service]
        )

    @pytest.mark.asyncio
    async def test_calculator_tool_integration(self, agent_worker):
        """Test that the calculator tool works correctly through the agent."""
        # Test basic operations
        result = await agent_worker._execute_tool("calculator", {
            "operation": "add",
            "a": 10,
            "b": 5
        })
        assert result == 15

        result = await agent_worker._execute_tool("calculator", {
            "operation": "multiply",
            "a": 4,
            "b": 7
        })
        assert result == 28

        result = await agent_worker._execute_tool("calculator", {
            "operation": "divide",
            "a": 20,
            "b": 4
        })
        assert result == 5.0

        # Test error handling
        with pytest.raises(ValueError, match="Cannot divide by zero"):
            await agent_worker._execute_tool("calculator", {
                "operation": "divide",
                "a": 10,
                "b": 0
            })

    @pytest.mark.asyncio
    async def test_weather_service_integration(self, agent_worker):
        """Test that the weather service tool works correctly through the agent."""
        # Test known cities
        result = await agent_worker._execute_tool("weather_service", {
            "city": "New York"
        })
        assert result == "Sunny, 25°C"

        result = await agent_worker._execute_tool("weather_service", {
            "city": "London"
        })
        assert result == "Rainy, 15°C"

        # Test unknown city
        result = await agent_worker._execute_tool("weather_service", {
            "city": "Unknown City"
        })
        assert result == "Weather data not available"

    @pytest.mark.asyncio
    async def test_tool_setup_integration(self, agent_worker):
        """Test that tools are properly set up for OpenAI API."""
        tools = agent_worker._setup_tools()
        assert len(tools) == 2

        # Check calculator tool schema
        calculator_tool = next(t for t in tools if t["function"]["name"] == "calculator")
        assert calculator_tool["type"] == "function"
        assert "operation" in calculator_tool["function"]["parameters"]["properties"]
        assert "a" in calculator_tool["function"]["parameters"]["properties"]
        assert "b" in calculator_tool["function"]["parameters"]["properties"]

        # Check weather service tool schema
        weather_tool = next(t for t in tools if t["function"]["name"] == "weather_service")
        assert weather_tool["type"] == "function"
        assert "city" in weather_tool["function"]["parameters"]["properties"]

    @pytest.mark.asyncio
    async def test_task_execution_with_tools(self, agent_worker, mock_client):
        """Test complete task execution flow with tool usage."""
        # Create a task that requires tool usage
        task = Task(
            id="integration_task_001",
            task_name="Calculate and get weather",
            goal="Calculate 15 + 7 and get weather for New York",
            result="",
            agent_name="integration_worker",
            context={"numbers": [15, 7], "city": "New York"}
        )

        # Mock the API responses to simulate tool usage
        mock_response1 = MagicMock()
        mock_response1.choices = [MagicMock()]
        mock_response1.choices[0].message.content = None
        mock_response1.choices[0].message.tool_calls = [
            {
                "id": "call_1",
                "name": "calculator",
                "arguments": '{"operation": "add", "a": 15, "b": 7}'
            }
        ]

        mock_response2 = MagicMock()
        mock_response2.choices = [MagicMock()]
        mock_response2.choices[0].message.content = None
        mock_response2.choices[0].message.tool_calls = [
            {
                "id": "call_2",
                "name": "weather_service",
                "arguments": '{"city": "New York"}'
            }
        ]

        mock_response3 = MagicMock()
        mock_response3.choices = [MagicMock()]
        mock_response3.choices[0].message.content = "The sum of 15 and 7 is 22. The weather in New York is Sunny, 25°C."
        mock_response3.choices[0].message.tool_calls = None

        mock_client.chat.completions.create.side_effect = [
            mock_response1, mock_response2, mock_response3
        ]

        # Execute the task
        result = await agent_worker.run(task)

        # Verify the task was completed successfully
        assert result.status == "success"
        assert "22" in result.result
        assert "Sunny, 25°C" in result.result

        # Verify the API was called 3 times (initial + 2 tool calls + final response)
        assert mock_client.chat.completions.create.call_count == 3

    @pytest.mark.asyncio
    async def test_agent_state_management(self, agent_worker):
        """Test that the agent properly manages its state."""
        # Check initial state
        initial_state = agent_worker.state
        assert initial_state["assigned_tasks"] is None
        assert len(initial_state["success_tasks"]) == 0
        assert len(initial_state["failed_tasks"]) == 0
        assert initial_state["retries"] == 0

        # Update state
        new_state = {
            "assigned_tasks": None,
            "success_tasks": [],
            "failed_tasks": [],
            "retries": 3
        }
        agent_worker.state = new_state

        # Verify state was updated
        updated_state = agent_worker.state
        assert updated_state["retries"] == 3

    @pytest.mark.asyncio
    async def test_tool_registry_functionality(self, agent_worker):
        """Test that the tool registry works correctly."""
        # Check that tools are properly registered
        assert "calculator" in agent_worker.tools_registry
        assert "weather_service" in agent_worker.tools_registry

        # Verify tool functions are accessible
        calc_func = agent_worker.tools_registry["calculator"]
        weather_func = agent_worker.tools_registry["weather_service"]

        # Test direct function calls
        assert calc_func("add", 5, 3) == 8
        assert weather_func("London") == "Rainy, 15°C"


if __name__ == "__main__":
    # Run the integration tests
    pytest.main([__file__, "-v"]) 