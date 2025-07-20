import pytest
import asyncio
from unittest.mock import Mock, AsyncMock, patch
from openai import AsyncOpenAI
from core.agent import (
    AgentRunner,
    PlanningAgent,
    AgentWorker,
    Task,
    TaskList,
    AgentState,
    ReviewResult,
)


class TestAgentRuntime:
    """Test suite for AgentRunner (AgentRuntime) functionality."""

    @pytest.fixture
    def mock_client(self):
        """Create a mock OpenAI client."""
        return Mock(spec=AsyncOpenAI)

    @pytest.fixture
    def mock_planning_agent(self, mock_client):
        """Create a mock planning agent."""
        agent = PlanningAgent(
            name="test_planner",
            client=mock_client,
            model="test-model"
        )
        return agent

    @pytest.fixture
    def mock_worker(self, mock_client):
        """Create a mock worker agent."""
        def dummy_tool(x: int) -> int:
            return x * 2
        
        worker = AgentWorker(
            name="test_worker",
            description="Test worker for unit tests",
            model="test-model",
            client=mock_client,
            system_prompt="You are a test worker.",
            tools=[dummy_tool]
        )
        return worker

    @pytest.fixture
    def sample_task(self):
        """Create a sample task for testing."""
        return Task(
            id="task-1",
            task_name="Test Task",
            goal="Test the functionality",
            result="",
            agent_name="test_worker",
            context={"test": "data"}
        )

    @pytest.fixture
    def sample_task_list(self, sample_task):
        """Create a sample task list for testing."""
        return TaskList(
            sequential_tasks=[sample_task],
            parallel_tasks=[]
        )

    @pytest.fixture
    def agent_runner(self, mock_planning_agent, mock_worker):
        """Create an AgentRunner instance for testing."""
        workers = {"test_worker": mock_worker}
        return AgentRunner(mock_planning_agent, workers)

    @pytest.mark.asyncio
    async def test_agent_runner_initialization(self, agent_runner, mock_planning_agent, mock_worker):
        """Test AgentRunner initialization."""
        assert agent_runner.planner == mock_planning_agent
        assert agent_runner.workers == {"test_worker": mock_worker}
        assert agent_runner.running is True
        assert agent_runner.task_queue is not None
        assert agent_runner.result_queue is not None

    @pytest.mark.asyncio
    async def test_dispatch_task_success(self, agent_runner, sample_task, mock_worker):
        """Test successful task dispatch."""
        # Mock the worker's run method
        mock_worker.run = AsyncMock(return_value=sample_task)
        
        await agent_runner.dispatch_task(sample_task)
        
        # Verify the worker was called
        mock_worker.run.assert_called_once_with(sample_task)

    @pytest.mark.asyncio
    async def test_dispatch_task_agent_not_found(self, agent_runner, sample_task):
        """Test task dispatch with non-existent agent."""
        # Create a task with non-existent agent
        task_with_wrong_agent = Task(
            id="task-2",
            task_name="Test Task",
            goal="Test the functionality",
            result="",
            agent_name="non_existent_worker",
            context={}
        )
        
        with pytest.raises(KeyError, match="non_existent_worker"):
            await agent_runner.dispatch_task(task_with_wrong_agent)

    @pytest.mark.asyncio
    async def test_execute_parallel_tasks(self, agent_runner, sample_task, mock_worker):
        """Test parallel task execution."""
        # Create multiple tasks
        task1 = Task(
            id="task-1",
            task_name="Task 1",
            goal="Goal 1",
            result="",
            agent_name="test_worker",
            context={}
        )
        task2 = Task(
            id="task-2",
            task_name="Task 2",
            goal="Goal 2",
            result="",
            agent_name="test_worker",
            context={}
        )
        
        # Mock the worker's run method
        mock_worker.run = AsyncMock(side_effect=[task1, task2])
        
        tasks = [task1, task2]
        await agent_runner.execute_parallel(tasks)
        
        # Verify both tasks were executed
        assert mock_worker.run.call_count == 2
        mock_worker.run.assert_any_call(task1)
        mock_worker.run.assert_any_call(task2)

    @pytest.mark.asyncio
    async def test_execute_sequential_tasks(self, agent_runner, sample_task, mock_worker):
        """Test sequential task execution."""
        # Create multiple tasks
        task1 = Task(
            id="task-1",
            task_name="Task 1",
            goal="Goal 1",
            result="",
            agent_name="test_worker",
            context={}
        )
        task2 = Task(
            id="task-2",
            task_name="Task 2",
            goal="Goal 2",
            result="",
            agent_name="test_worker",
            context={}
        )
        
        # Mock the worker's run method
        mock_worker.run = AsyncMock(side_effect=[task1, task2])
        
        tasks = [task1, task2]
        await agent_runner.execute_sequential(tasks)
        
        # Verify tasks were executed in order
        assert mock_worker.run.call_count == 2
        mock_worker.run.assert_any_call(task1)
        mock_worker.run.assert_any_call(task2)

    @pytest.mark.asyncio
    async def test_handle_result_loop_success(self, agent_runner, sample_task, mock_planning_agent):
        """Test result handling loop with successful task."""
        # Mock the review result
        review_result = ReviewResult(is_success="success", feedback="Good job!")
        mock_planning_agent._review_single_task = AsyncMock(return_value=review_result)
        
        # Add a task to the result queue
        await agent_runner.result_queue.put(sample_task)
        
        # Start the result loop
        result_loop_task = asyncio.create_task(agent_runner.handle_result_loop())
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Stop the loop
        agent_runner.running = False
        await result_loop_task
        
        # Verify the review was called
        mock_planning_agent._review_single_task.assert_called_once_with(sample_task)
        
        # Verify task was added to success list
        assert sample_task in mock_planning_agent._state.success_tasks

    @pytest.mark.asyncio
    async def test_handle_result_loop_failure(self, agent_runner, sample_task, mock_planning_agent):
        """Test result handling loop with failed task."""
        # Mock the review result
        review_result = ReviewResult(is_success="failure", feedback="Task failed")
        mock_planning_agent._review_single_task = AsyncMock(return_value=review_result)
        
        # Add a task to the result queue
        await agent_runner.result_queue.put(sample_task)
        
        # Start the result loop
        result_loop_task = asyncio.create_task(agent_runner.handle_result_loop())
        
        # Wait a bit for processing
        await asyncio.sleep(0.1)
        
        # Stop the loop
        agent_runner.running = False
        await result_loop_task
        
        # Verify the review was called
        mock_planning_agent._review_single_task.assert_called_once_with(sample_task)
        
        # Verify task was added to failed list and status updated
        assert sample_task in mock_planning_agent._state.failed_tasks
        assert sample_task.status == "failure"

    @pytest.mark.asyncio
    async def test_run_complete_workflow(self, agent_runner, sample_task_list, mock_planning_agent, mock_worker):
        """Test the complete workflow from query to final state."""
        # Mock planning
        mock_planning_agent.plan = AsyncMock(return_value=sample_task_list)
        
        # Mock worker execution
        completed_task = sample_task_list.sequential_tasks[0]
        completed_task.status = "success"
        completed_task.result = "Task completed successfully"
        mock_worker.run = AsyncMock(return_value=completed_task)
        
        # Mock review
        review_result = ReviewResult(is_success="success", feedback="Good job!")
        mock_planning_agent._review_single_task = AsyncMock(return_value=review_result)
        
        # Run the workflow
        result = await agent_runner.run("Test query")
        
        # Verify planning was called
        mock_planning_agent.plan.assert_called_once_with("Test query")
        
        # Verify worker was called
        mock_worker.run.assert_called_once_with(sample_task_list.sequential_tasks[0])
        
        # Verify review was called
        mock_planning_agent._review_single_task.assert_called_once_with(completed_task)
        
        # Verify final state
        assert isinstance(result, AgentState)
        assert completed_task in result.success_tasks

    @pytest.mark.asyncio
    async def test_run_with_parallel_and_sequential_tasks(self, agent_runner, mock_planning_agent, mock_worker):
        """Test running with both parallel and sequential tasks."""
        # Create task list with both parallel and sequential tasks
        parallel_task = Task(
            id="parallel-1",
            task_name="Parallel Task",
            goal="Parallel goal",
            result="",
            agent_name="test_worker",
            context={}
        )
        sequential_task = Task(
            id="sequential-1",
            task_name="Sequential Task",
            goal="Sequential goal",
            result="",
            agent_name="test_worker",
            context={}
        )
        
        task_list = TaskList(
            sequential_tasks=[sequential_task],
            parallel_tasks=[parallel_task]
        )
        
        # Mock planning
        mock_planning_agent.plan = AsyncMock(return_value=task_list)
        
        # Mock worker execution
        completed_parallel = parallel_task.model_copy()
        completed_parallel.status = "success"
        completed_parallel.result = "Parallel completed"
        
        completed_sequential = sequential_task.model_copy()
        completed_sequential.status = "success"
        completed_sequential.result = "Sequential completed"
        
        mock_worker.run = AsyncMock(side_effect=[completed_parallel, completed_sequential])
        
        # Mock review
        review_result = ReviewResult(is_success="success", feedback="Good job!")
        mock_planning_agent._review_single_task = AsyncMock(return_value=review_result)
        
        # Run the workflow
        result = await agent_runner.run("Test query with mixed tasks")
        
        # Verify both tasks were executed
        assert mock_worker.run.call_count == 2
        
        # Verify final state contains both tasks
        assert len(result.success_tasks) == 2

    @pytest.mark.asyncio
    async def test_run_with_empty_task_list(self, agent_runner, mock_planning_agent):
        """Test running with empty task list."""
        # Create empty task list
        empty_task_list = TaskList(sequential_tasks=[], parallel_tasks=[])
        
        # Mock planning
        mock_planning_agent.plan = AsyncMock(return_value=empty_task_list)
        
        # Run the workflow
        result = await agent_runner.run("Empty query")
        
        # Verify planning was called
        mock_planning_agent.plan.assert_called_once_with("Empty query")
        
        # Verify final state is empty
        assert len(result.success_tasks) == 0
        assert len(result.failed_tasks) == 0

    @pytest.mark.asyncio
    async def test_agent_runner_state_management(self, agent_runner):
        """Test that AgentRunner properly manages its running state."""
        assert agent_runner.running is True
        
        # Test state change during run
        agent_runner.running = False
        assert agent_runner.running is False
        
        # Reset for next test
        agent_runner.running = True

    @pytest.mark.asyncio
    async def test_queue_operations(self, agent_runner, sample_task):
        """Test queue operations in AgentRunner."""
        # Test putting and getting from result queue
        await agent_runner.result_queue.put(sample_task)
        retrieved_task = await agent_runner.result_queue.get()
        
        assert retrieved_task == sample_task
        assert retrieved_task.id == sample_task.id
        assert retrieved_task.task_name == sample_task.task_name
