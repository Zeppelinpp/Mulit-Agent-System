import inspect
import json
from typing import Any, Callable, Dict, List, Optional, Type, Union, get_type_hints
from pydantic import BaseModel, create_model


class ToolConverter:
    """Converter for callable functions to OpenAI tools format."""
    
    @staticmethod
    def convert_callable_to_tool(
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert a callable function to OpenAI tool format.
        
        Args:
            func: The callable function to convert
            name: Optional custom name for the tool
            description: Optional custom description for the tool
            
        Returns:
            Dictionary in OpenAI tool format
        """
        # Get function metadata
        func_name = name or func.__name__
        func_description = description or func.__doc__ or ""
        
        # Get type hints
        type_hints = get_type_hints(func)
        
        # Extract parameters (excluding 'self' for methods)
        sig = inspect.signature(func)
        parameters = {}
        required_params = []
        
        for param_name, param in sig.parameters.items():
            # Skip 'self' parameter for methods
            if param_name == 'self':
                continue
                
            param_type = type_hints.get(param_name, Any)
            param_default = param.default
            
            # Determine if parameter is required
            is_required = param.default == inspect.Parameter.empty
            
            if is_required:
                required_params.append(param_name)
            
            # Convert parameter to JSON schema
            param_schema = ToolConverter._convert_type_to_schema(param_type)
            
            # Add default value if present
            if param_default != inspect.Parameter.empty:
                param_schema["default"] = param_default
            
            parameters[param_name] = param_schema
        
        # Create the tool schema
        tool_schema = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func_description,
                "parameters": {
                    "type": "object",
                    "properties": parameters,
                    "required": required_params
                }
            }
        }
        
        return tool_schema
    
    @staticmethod
    def _convert_type_to_schema(type_hint: Type) -> Dict[str, Any]:
        """
        Convert a Python type hint to JSON schema format.
        
        Args:
            type_hint: The type hint to convert
            
        Returns:
            JSON schema dictionary
        """
        # Handle Pydantic BaseModel
        if isinstance(type_hint, type) and issubclass(type_hint, BaseModel):
            return type_hint.model_json_schema()
        
        # Handle Union types (including Optional)
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is Union:
            # Handle Optional[T] which is Union[T, None]
            args = type_hint.__args__
            if len(args) == 2 and type(None) in args:
                # This is Optional[T]
                non_none_type = next(arg for arg in args if arg is not type(None))
                return ToolConverter._convert_type_to_schema(non_none_type)
            else:
                # This is Union[T1, T2, ...]
                return {"oneOf": [ToolConverter._convert_type_to_schema(arg) for arg in args]}
        
        # Handle List types
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is list:
            args = type_hint.__args__
            if args:
                return {
                    "type": "array",
                    "items": ToolConverter._convert_type_to_schema(args[0])
                }
            else:
                return {"type": "array"}
        
        # Handle Dict types
        if hasattr(type_hint, "__origin__") and type_hint.__origin__ is dict:
            args = type_hint.__args__
            if len(args) >= 2:
                return {
                    "type": "object",
                    "additionalProperties": ToolConverter._convert_type_to_schema(args[1])
                }
            else:
                return {"type": "object"}
        
        # Handle basic types
        type_mapping = {
            str: {"type": "string"},
            int: {"type": "integer"},
            float: {"type": "number"},
            bool: {"type": "boolean"},
            list: {"type": "array"},
            dict: {"type": "object"},
        }
        
        if type_hint in type_mapping:
            return type_mapping[type_hint]
        
        # Handle Any type
        if type_hint == Any:
            return {}
        
        # Default fallback
        return {"type": "string"}


class PydanticToolConverter:
    """Specialized converter for functions with Pydantic input/output models."""
    
    @staticmethod
    def convert_with_pydantic_models(
        func: Callable,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None,
        name: Optional[str] = None,
        description: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Convert a callable function to OpenAI tool format with explicit Pydantic models.
        
        Args:
            func: The callable function to convert
            input_model: Pydantic model for input parameters
            output_model: Pydantic model for output
            name: Optional custom name for the tool
            description: Optional custom description for the tool
            
        Returns:
            Dictionary in OpenAI tool format
        """
        func_name = name or func.__name__
        func_description = description or func.__doc__ or ""
        
        # Use provided input model or infer from function signature
        if input_model is None:
            input_model = PydanticToolConverter._infer_input_model(func)
        
        # Create tool schema
        tool_schema = {
            "type": "function",
            "function": {
                "name": func_name,
                "description": func_description,
                "parameters": input_model.model_json_schema() if input_model else {
                    "type": "object",
                    "properties": {},
                    "required": []
                }
            }
        }
        
        return tool_schema
    
    @staticmethod
    def _infer_input_model(func: Callable) -> Optional[Type[BaseModel]]:
        """
        Infer input Pydantic model from function signature.
        
        Args:
            func: The function to analyze
            
        Returns:
            Pydantic model class or None if no model found
        """
        sig = inspect.signature(func)
        type_hints = get_type_hints(func)
        
        # Check if any parameter is a Pydantic model
        pydantic_params = {}
        for param_name, param in sig.parameters.items():
            if param_name == 'self':
                continue
                
            param_type = type_hints.get(param_name)
            if isinstance(param_type, type) and issubclass(param_type, BaseModel):
                # If a parameter is already a Pydantic model, use it directly
                return param_type
            elif param_type is not None:
                # Convert other types to schema
                pydantic_params[param_name] = (param_type, ...)
        
        # Create a new Pydantic model if we have parameters
        if pydantic_params:
            return create_model(f"{func.__name__}Input", **pydantic_params)
        
        return None


class ToolRegistry:
    """Registry for managing tools and their conversions."""
    
    def __init__(self):
        self._tools: Dict[str, Callable] = {}
        self._tool_schemas: Dict[str, Dict[str, Any]] = {}
    
    def register_tool(
        self,
        func: Callable,
        name: Optional[str] = None,
        description: Optional[str] = None,
        input_model: Optional[Type[BaseModel]] = None,
        output_model: Optional[Type[BaseModel]] = None
    ) -> str:
        """
        Register a tool function.
        
        Args:
            func: The function to register
            name: Optional custom name
            description: Optional custom description
            input_model: Optional Pydantic input model
            output_model: Optional Pydantic output model
            
        Returns:
            The registered tool name
        """
        tool_name = name or func.__name__
        
        # Convert to OpenAI tool format
        if input_model or output_model:
            tool_schema = PydanticToolConverter.convert_with_pydantic_models(
                func, input_model, output_model, name, description
            )
        else:
            tool_schema = ToolConverter.convert_callable_to_tool(func, name, description)
        
        self._tools[tool_name] = func
        self._tool_schemas[tool_name] = tool_schema
        
        return tool_name
    
    def get_tool(self, name: str) -> Optional[Callable]:
        """Get a registered tool by name."""
        return self._tools.get(name)
    
    def get_tool_schema(self, name: str) -> Optional[Dict[str, Any]]:
        """Get the OpenAI tool schema for a registered tool."""
        return self._tool_schemas.get(name)
    
    def get_all_tool_schemas(self) -> List[Dict[str, Any]]:
        """Get all registered tool schemas."""
        return list(self._tool_schemas.values())
    
    def execute_tool(self, name: str, **kwargs) -> Any:
        """Execute a registered tool with given arguments."""
        tool = self.get_tool(name)
        if tool is None:
            raise ValueError(f"Tool '{name}' not found")
        return tool(**kwargs)


# Convenience functions
def convert_to_openai_tool(
    func: Callable,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Convert a callable function to OpenAI tool format."""
    return ToolConverter.convert_callable_to_tool(func, name, description)


def convert_with_pydantic_models(
    func: Callable,
    input_model: Optional[Type[BaseModel]] = None,
    output_model: Optional[Type[BaseModel]] = None,
    name: Optional[str] = None,
    description: Optional[str] = None
) -> Dict[str, Any]:
    """Convert a callable function to OpenAI tool format with Pydantic models."""
    return PydanticToolConverter.convert_with_pydantic_models(
        func, input_model, output_model, name, description
    )
