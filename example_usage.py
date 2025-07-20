from pydantic import BaseModel, Field
from typing import List, Optional
from core.utils import (
    ToolConverter, 
    PydanticToolConverter, 
    ToolRegistry,
    convert_to_openai_tool,
    convert_with_pydantic_models
)


# Example Pydantic models
class UserInput(BaseModel):
    name: str = Field(..., description="User's full name")
    age: int = Field(..., ge=0, le=120, description="User's age")
    email: Optional[str] = Field(None, description="User's email address")


class UserOutput(BaseModel):
    user_id: str = Field(..., description="Generated user ID")
    greeting: str = Field(..., description="Personalized greeting message")
    status: str = Field(..., description="Account status")


class SearchInput(BaseModel):
    query: str = Field(..., description="Search query")
    limit: int = Field(default=10, ge=1, le=100, description="Maximum number of results")


class SearchResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchOutput(BaseModel):
    results: List[SearchResult] = Field(..., description="Search results")
    total_count: int = Field(..., description="Total number of results found")


# Example functions
def create_user(name: str, age: int, email: Optional[str] = None) -> UserOutput:
    """Create a new user account."""
    user_id = f"user_{name.lower().replace(' ', '_')}_{age}"
    greeting = f"Hello {name}! Welcome to our platform."
    status = "active"
    
    return UserOutput(user_id=user_id, greeting=greeting, status=status)


def search_web(query: str, limit: int = 10) -> SearchOutput:
    """Search the web for information."""
    # Simulate search results
    results = [
        SearchResult(
            title=f"Result {i} for {query}",
            url=f"https://example.com/result{i}",
            snippet=f"This is a sample result {i} for the query '{query}'"
        )
        for i in range(1, min(limit + 1, 6))
    ]
    
    return SearchOutput(results=results, total_count=len(results))


def calculate_area(length: float, width: float) -> float:
    """Calculate the area of a rectangle."""
    return length * width


def process_text(text: str, uppercase: bool = False) -> str:
    """Process text with optional uppercase conversion."""
    if uppercase:
        return text.upper()
    return text


def main():
    print("=== Tool Converter Examples ===\n")
    
    # Example 1: Basic function conversion
    print("1. Basic function conversion:")
    area_tool = convert_to_openai_tool(calculate_area)
    print(json.dumps(area_tool, indent=2))
    print()
    
    # Example 2: Function with Pydantic models
    print("2. Function with Pydantic models:")
    user_tool = convert_with_pydantic_models(
        create_user,
        input_model=UserInput,
        output_model=UserOutput
    )
    print(json.dumps(user_tool, indent=2))
    print()
    
    # Example 3: Using ToolRegistry
    print("3. Using ToolRegistry:")
    registry = ToolRegistry()
    
    # Register tools
    registry.register_tool(calculate_area, description="Calculate rectangle area")
    registry.register_tool(
        search_web,
        input_model=SearchInput,
        output_model=SearchOutput,
        description="Search the web for information"
    )
    registry.register_tool(process_text, description="Process text with options")
    
    # Get all tool schemas
    all_tools = registry.get_all_tool_schemas()
    print(f"Registered {len(all_tools)} tools:")
    for tool in all_tools:
        print(f"- {tool['function']['name']}: {tool['function']['description']}")
    print()
    
    # Example 4: Execute a tool
    print("4. Executing a tool:")
    result = registry.execute_tool("calculate_area", length=5.0, width=3.0)
    print(f"Area calculation result: {result}")
    print()
    
    # Example 5: Function with complex types
    print("5. Function with complex types:")
    def analyze_data(data: List[dict], threshold: float = 0.5) -> dict:
        """Analyze a list of data points."""
        return {
            "count": len(data),
            "above_threshold": len([d for d in data if d.get("value", 0) > threshold]),
            "average": sum(d.get("value", 0) for d in data) / len(data) if data else 0
        }
    
    analyze_tool = convert_to_openai_tool(analyze_data)
    print(json.dumps(analyze_tool, indent=2))


if __name__ == "__main__":
    import json
    main() 