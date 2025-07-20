# Multi-Agent System

A multi-agent system with planning and execution capabilities, featuring a powerful tool converter for OpenAI function calling.

## Features

- **Planning Agent**: Breaks down complex tasks into sequential and parallel subtasks
- **Agent Workers**: Execute specific tasks with tool support
- **Tool Converter**: Convert Python callables to OpenAI tools format with Pydantic model support
- **Tool Registry**: Manage and execute registered tools

## Tool Converter

The tool converter (`core/utils.py`) provides comprehensive functionality to convert Python callable functions to OpenAI tools format, with full support for Pydantic BaseModel inputs and outputs.

### Basic Usage

```python
from core.utils import convert_to_openai_tool

def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width

# Convert to OpenAI tool format
tool_schema = convert_to_openai_tool(calculate_area)
```

### Pydantic Model Support

```python
from pydantic import BaseModel, Field
from core.utils import convert_with_pydantic_models

class UserInput(BaseModel):
    name: str = Field(..., description="User's full name")
    age: int = Field(..., ge=0, le=120, description="User's age")

class UserOutput(BaseModel):
    user_id: str
    greeting: str

def create_user(name: str, age: int) -> UserOutput:
    """Create a new user account."""
    return UserOutput(user_id=f"user_{name}", greeting=f"Hello {name}!")

# Convert with explicit Pydantic models
tool_schema = convert_with_pydantic_models(
    create_user,
    input_model=UserInput,
    output_model=UserOutput
)
```

### Tool Registry

```python
from core.utils import ToolRegistry

registry = ToolRegistry()

# Register tools
registry.register_tool(calculate_area, description="Calculate rectangle area")
registry.register_tool(
    create_user,
    input_model=UserInput,
    output_model=UserOutput,
    description="Create a new user"
)

# Get all tool schemas for OpenAI
tools = registry.get_all_tool_schemas()

# Execute a tool
result = registry.execute_tool("calculate_area", length=5.0, width=3.0)
```

### Supported Types

The converter supports:

- **Basic Types**: `str`, `int`, `float`, `bool`, `list`, `dict`
- **Complex Types**: `List[T]`, `Dict[K, V]`, `Optional[T]`, `Union[T1, T2]`
- **Pydantic Models**: Full support for BaseModel classes
- **Default Values**: Automatically detected from function signatures
- **Required Parameters**: Determined from function signatures

### Advanced Features

1. **Automatic Type Inference**: Converts Python type hints to JSON schema
2. **Pydantic Integration**: Seamless support for Pydantic validation and serialization
3. **Method Support**: Handles class methods (skips `self` parameter)
4. **Flexible Naming**: Custom tool names and descriptions
5. **Registry Management**: Centralized tool management and execution

## Installation

```bash
# Install dependencies
pip install -e .
```

## Usage Examples

See `example_usage.py` for comprehensive examples of using the tool converter.

## Project Structure

```
multi-agent-system/
├── core/
│   ├── agent.py          # Agent implementations
│   ├── agent_prompts.py  # System prompts
│   └── utils.py          # Tool converter and utilities
├── main.py               # Main application entry point
├── example_usage.py      # Tool converter examples
├── pyproject.toml        # Project configuration
└── README.md            # This file
```

## License

MIT License
