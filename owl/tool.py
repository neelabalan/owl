import abc
import asyncio
import dataclasses
import enum
import json
import logging
import typing


class ToolType(enum.Enum):
    FUNCTION = 'function'
    ASYNC_FUNCTION = 'async_function'
    MCP_TOOL = 'mcp_tool'
    API_ENDPOINT = 'api_endpoint'


@dataclasses.dataclass
class ToolParameter:
    name: str
    type_hint: str
    required: bool = True
    description: str = ''
    default: typing.Any = None


@dataclasses.dataclass
class ToolSchema:
    name: str
    description: str
    parameters: list[ToolParameter]
    tool_type: ToolType
    metadata: dict[str, typing.Any] = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}

    def format_for_llm(self) -> str:
        args_desc = []
        for param in self.parameters:
            arg_desc = f'- {param.name}: {param.type_hint}'
            if param.description:
                arg_desc += f' - {param.description}'
            if param.required:
                arg_desc += ' (required)'
            elif param.default is not None:
                arg_desc += f' (default: {param.default})'
            args_desc.append(arg_desc)

        required_fields = [p.name for p in self.parameters if p.required]
        if required_fields:
            args_desc.append(f'\nRequired fields: {", ".join(required_fields)}')

        import textwrap

        return textwrap.dedent(f"""
            Tool: {self.name}
            Description: {self.description}
            Arguments:
            {chr(10).join(args_desc)}
            """).strip()


class ToolExecutor(abc.ABC):
    @abc.abstractmethod
    async def execute(self, arguments: dict[str, typing.Any]) -> typing.Any:
        pass

    @abc.abstractmethod
    def get_schema(self) -> ToolSchema:
        pass


class FunctionToolExecutor(ToolExecutor):
    def __init__(self, func: typing.Union[typing.Callable, typing.Callable[..., typing.Awaitable]], schema: ToolSchema):
        self.func = func
        self.schema = schema
        self.is_async = asyncio.iscoroutinefunction(func)

    async def execute(self, arguments: dict[str, typing.Any]) -> typing.Any:
        try:
            if self.is_async:
                return await self.func(**arguments)
            else:
                return self.func(**arguments)
        except TypeError as e:
            raise ToolExecutionError(f'Invalid arguments for {self.schema.name}: {e}')

    def get_schema(self) -> ToolSchema:
        return self.schema


class MCPToolExecutor(ToolExecutor):
    def __init__(self, tool_name: str, server_session: typing.Any, schema: ToolSchema):
        self.tool_name = tool_name
        self.server_session = server_session
        self.schema = schema

    async def execute(self, arguments: dict[str, typing.Any]) -> typing.Any:
        try:
            result = await self.server_session.call_tool(self.tool_name, {'request': arguments})
            return result
        except Exception as e:
            raise ToolExecutionError(f'MCP tool execution failed: {e}')

    def get_schema(self) -> ToolSchema:
        return self.schema


class ToolExecutionError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolExecutor] = {}

    def register_tool(self, tool_executor: ToolExecutor) -> None:
        schema = tool_executor.get_schema()
        self._tools[schema.name] = tool_executor

    def register_function(
        self,
        func: typing.Union[typing.Callable, typing.Callable[..., typing.Awaitable]],
        name: str = None,
        description: str = '',
        parameters: list[ToolParameter] = None,
    ) -> None:
        if name is None:
            name = func.__name__

        if parameters is None:
            parameters = self._infer_parameters_from_function(func)

        tool_type = ToolType.ASYNC_FUNCTION if asyncio.iscoroutinefunction(func) else ToolType.FUNCTION
        schema = ToolSchema(name, description, parameters, tool_type)
        executor = FunctionToolExecutor(func, schema)
        self.register_tool(executor)

    def _infer_parameters_from_function(self, func: typing.Callable) -> list[ToolParameter]:
        import inspect

        signature = inspect.signature(func)
        parameters = []

        for param_name, param in signature.parameters.items():
            type_hint = str(param.annotation) if param.annotation != inspect.Parameter.empty else 'Any'
            required = param.default == inspect.Parameter.empty
            default = param.default if param.default != inspect.Parameter.empty else None

            doc_description = ''
            if func.__doc__:
                lines = func.__doc__.split('\n')
                for line in lines:
                    if param_name in line and ':' in line:
                        doc_description = line.split(':', 1)[1].strip()
                        break

            parameters.append(
                ToolParameter(
                    name=param_name, type_hint=type_hint, required=required, description=doc_description, default=default
                )
            )

        return parameters

    def get_tool(self, name: str) -> ToolExecutor:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_tools(self) -> list[ToolSchema]:
        return [tool.get_schema() for tool in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools


class ToolCallParser:
    @staticmethod
    def parse_tool_call(response: str) -> dict[str, typing.Any] | None:
        try:
            parsed = json.loads(response.strip())
            if isinstance(parsed, dict) and 'tool' in parsed and 'arguments' in parsed:
                return parsed
            return None
        except json.JSONDecodeError:
            return None

    @staticmethod
    def format_tool_call(tool_name: str, arguments: dict[str, typing.Any]) -> str:
        return json.dumps({'tool': tool_name, 'arguments': arguments}, indent=2)


class ToolCallResult:
    def __init__(self, success: bool, result: typing.Any = None, error: str = None):
        self.success = success
        self.result = result
        self.error = error

    def to_dict(self) -> dict[str, typing.Any]:
        return {'success': self.success, 'result': self.result, 'error': self.error}


class UniversalToolCaller:
    def __init__(self, registry: ToolRegistry):
        self.registry = registry
        self.parser = ToolCallParser()

    async def execute_tool_call(self, response: str) -> ToolCallResult:
        tool_call = self.parser.parse_tool_call(response)
        if not tool_call:
            return ToolCallResult(success=False, error='Invalid tool call format')

        tool_name = tool_call['tool']
        arguments = tool_call['arguments']

        try:
            if not self.registry.has_tool(tool_name):
                return ToolCallResult(success=False, error=f"Tool '{tool_name}' not found")

            tool_executor = self.registry.get_tool(tool_name)
            result = await tool_executor.execute(arguments)
            return ToolCallResult(success=True, result=result)

        except ToolExecutionError as e:
            return ToolCallResult(success=False, error=str(e))
        except Exception as e:
            logging.error(f'Unexpected error executing tool {tool_name}: {e}')
            return ToolCallResult(success=False, error=f'Unexpected error: {e}')


def format_tools_for_prompt(tools: list[ToolSchema]) -> str:
    """Format tool schemas for use in prompts"""
    return '\n'.join([tool.format_for_llm() for tool in tools])


def create_tool_from_mcp(
    tool_name: str, description: str, input_schema: dict[str, typing.Any], server_session: typing.Any
) -> MCPToolExecutor:
    parameters = []

    if '$defs' in input_schema and input_schema['$defs']:
        key = list(input_schema['$defs'].keys())[0]
        properties = input_schema['$defs'][key].get('properties', {})
        required_fields = input_schema['$defs'][key].get('required', [])

        for param_name, param_info in properties.items():
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type_hint=param_info.get('type', 'Any'),
                    required=param_name in required_fields,
                    description=param_info.get('description', ''),
                )
            )

    schema = ToolSchema(tool_name, description, parameters, ToolType.MCP_TOOL)
    return MCPToolExecutor(tool_name, server_session, schema)
