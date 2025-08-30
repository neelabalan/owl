import abc
import asyncio
import dataclasses
import enum
import inspect
import json
import logging
import typing

import pydantic

import owl.parameter_inference as param_infer
from owl.common import extract_function_description
from owl.jsonrpc import JsonRpcRequest
from owl.mcp_manager import MCPClient
from owl.mcp_manager import MCPServer

if typing.TYPE_CHECKING:
    from owl.mcp_manager import MCPServerManager


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

    @classmethod
    def from_function(
        cls,
        func: typing.Callable,
        name: str = None,
        description: str = '',
        parameters: list[ToolParameter] = None,
    ) -> 'ToolSchema':
        if name is None:
            name = func.__name__

        # Prefer docstring description over provided description
        docstring_description = extract_function_description(func)
        final_description = docstring_description if docstring_description else description

        if parameters is None:
            parameters = cls._infer_parameters_from_function(func)

        if asyncio.iscoroutinefunction(func):
            tool_type = ToolType.ASYNC_FUNCTION
        else:
            tool_type = ToolType.FUNCTION

        return cls(name, final_description, parameters, tool_type)

    @staticmethod
    def _infer_parameters_from_function(func: typing.Callable) -> list[ToolParameter]:
        parameter_engine = param_infer.ParameterInferenceEngine()
        parameter_formatter = param_infer.ParameterFormatter()
        param_infos = parameter_engine.infer_parameters(func)

        parameters = []
        for param_info in param_infos:
            # for complex parameters (like Pydantic models with nested fields),
            # use the formatter for description
            if param_info.nested_fields:
                # this is a complex parameter, format it properly
                enhanced_description = param_info.description
                if param_info.constraints:
                    constraint_str = parameter_formatter._format_constraints(param_info.constraints)
                    enhanced_description += f' {constraint_str}' if constraint_str else ''

                # create a special tool parameter that includes nested fields info
                tool_param = ToolParameter(
                    name=param_info.name,
                    type_hint=param_info.type_hint,
                    required=param_info.required,
                    description=enhanced_description,
                    default=param_info.default,
                )
                # attach nested fields for later formatting
                tool_param._nested_fields = param_info.nested_fields
                parameters.append(tool_param)
            else:
                # simple parameter, keep existing behavior
                enhanced_description = param_info.description
                if param_info.constraints:
                    constraint_str = parameter_formatter._format_constraints(param_info.constraints)
                    enhanced_description += f' {constraint_str}' if constraint_str else ''

                parameters.append(
                    ToolParameter(
                        name=param_info.name,
                        type_hint=param_info.type_hint,
                        required=param_info.required,
                        description=enhanced_description,
                        default=param_info.default,
                    )
                )

        return parameters

    def format_for_llm(self) -> str:
        # check if we have any complex parameters (with nested fields)
        has_complex_params = any(hasattr(param, '_nested_fields') and param._nested_fields for param in self.parameters)
        if has_complex_params:
            formatter = param_infer.ParameterFormatter()
            # convert ToolParameter back to ParameterInfo for formatting
            param_infos = []
            for param in self.parameters:
                param_info = param_infer.ParameterInfo(
                    name=param.name,
                    type_hint=param.type_hint,
                    required=param.required,
                    description=param.description,
                    default=param.default,
                )
                # add nested fields if available
                if hasattr(param, '_nested_fields'):
                    param_info.nested_fields = param._nested_fields
                param_infos.append(param_info)
            formatted_args = formatter.format_parameter_list(param_infos)
        else:
            # use simple formatting for basic parameters
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

            # Format arguments with proper indentation
            formatted_args = '\n'.join(f'  {arg}' for arg in args_desc)

        return f"""Tool: {self.name}\nDescription: {self.description}\nArguments:\n{formatted_args}\n\n"""


class ToolExecutor(abc.ABC):
    @abc.abstractmethod
    def execute(self, arguments: dict[str, typing.Any]) -> typing.Any:
        pass

    @abc.abstractmethod
    def get_schema(self) -> ToolSchema:
        pass

    @abc.abstractmethod
    def is_async(self) -> bool:
        pass


class FunctionToolExecutor(ToolExecutor):
    def __init__(self, func: typing.Callable, schema: ToolSchema):
        self.func = func
        self.schema = schema

    def execute(self, arguments: dict[str, typing.Any]) -> typing.Any:
        try:
            converted_args = _convert_pydantic_arguments(self.func, arguments)
            return self.func(**converted_args)
        except TypeError as e:
            raise ToolExecutionError(f'Invalid arguments for {self.schema.name}: {e}')
        except Exception as e:
            raise ToolExecutionError(f'Error converting arguments for {self.schema.name}: {e}')

    def get_schema(self) -> ToolSchema:
        return self.schema

    def is_async(self) -> bool:
        return False


class AsyncFunctionToolExecutor(ToolExecutor):
    def __init__(self, func: typing.Callable[..., typing.Awaitable], schema: ToolSchema):
        self.func = func
        self.schema = schema

    async def execute(self, arguments: dict[str, typing.Any]) -> typing.Any:
        try:
            converted_args = _convert_pydantic_arguments(self.func, arguments)
            return await self.func(**converted_args)
        except TypeError as e:
            raise ToolExecutionError(f'Invalid arguments for {self.schema.name}: {e}')
        except Exception as e:
            raise ToolExecutionError(f'Error converting arguments for {self.schema.name}: {e}')

    def get_schema(self) -> ToolSchema:
        return self.schema

    def is_async(self) -> bool:
        return True


class MCPToolExecutor(ToolExecutor):
    def __init__(self, tool_name: str, server: MCPServer, schema: ToolSchema):
        self.tool_name = tool_name
        self.server = server
        self.schema = schema

    async def execute(self, arguments: dict[str, typing.Any]) -> typing.Any:
        try:
            transport = self.server.get_transport()
            if not transport:
                raise ToolExecutionError(f'No transport available for server {self.server_name}')

            await transport.send(
                JsonRpcRequest(
                    jsonrpc='2.0', id=3, method='tools/call', params={'name': self.tool_name, 'arguments': arguments}
                ).model_dump_json(exclude_none=True)
            )
            response_line = await transport.receive()
            response = json.loads(response_line) if response_line else {}

            if response.get('result'):
                content = response['result'].get('content', [])
                if content and isinstance(content, list):
                    return '\n'.join([item.get('text', '') for item in content])
                return str(response['result'])
            elif response.get('error'):
                raise ToolExecutionError(f'MCP tool error: {response["error"]}')

            return 'No response from MCP tool'

        except Exception as e:
            raise ToolExecutionError(f'Error executing MCP tool {self.tool_name}: {e}')

    def get_schema(self) -> ToolSchema:
        return self.schema

    def is_async(self) -> bool:
        return True


class ToolExecutionError(Exception):
    pass


class ToolRegistry:
    def __init__(self):
        self._tools: dict[str, ToolExecutor] = {}

    def register(
        self,
        tool: typing.Union[ToolExecutor, typing.Callable, typing.Callable[..., typing.Awaitable]],
        name: str = None,
        description: str = '',
        parameters: list[ToolParameter] = None,
    ) -> None:
        if isinstance(tool, ToolExecutor):
            schema = tool.get_schema()
            self._tools[schema.name] = tool
        elif callable(tool):
            # Use ToolSchema to create schema from function
            schema = ToolSchema.from_function(tool, name, description, parameters)

            if asyncio.iscoroutinefunction(tool):
                executor = AsyncFunctionToolExecutor(tool, schema)
            else:
                executor = FunctionToolExecutor(tool, schema)

            self._tools[schema.name] = executor
        else:
            raise ValueError(f'Cannot register tool of type {type(tool)}')

    def get_tool(self, name: str) -> ToolExecutor:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self._tools[name]

    def list_tools(self) -> list[ToolSchema]:
        return [tool.get_schema() for tool in self._tools.values()]

    def has_tool(self, name: str) -> bool:
        return name in self._tools

    def is_tool_async(self, name: str) -> bool:
        if name not in self._tools:
            raise ValueError(f"Tool '{name}' not found in registry")
        return self._tools[name].is_async()


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

    def is_tool_call(self, response: str) -> bool:
        tool_call = self.parser.parse_tool_call(response)
        return tool_call is not None

    # figure out how to do this properly
    def run_async(self, func, *args, **kwargs):
        try:
            loop = asyncio.get_running_loop()
        except RuntimeError:
            loop = None

        def runner():
            return asyncio.run(func(*args, **kwargs))

        if loop and loop.is_running():
            import queue
            from threading import Thread

            q = queue.Queue()

            def thread_target():
                try:
                    q.put(runner())
                except Exception as e:
                    q.put(e)

            t = Thread(target=thread_target)
            t.start()
            t.join()
            result = q.get()
            if isinstance(result, Exception):
                raise result
            return result
        else:
            return asyncio.run(func(*args, **kwargs))

    def execute_tool_call(self, response: str) -> ToolCallResult:
        tool_call = self.parser.parse_tool_call(response)
        if not tool_call:
            return ToolCallResult(success=False, error='Invalid tool call format')

        tool_name = tool_call['tool']
        arguments = tool_call['arguments']

        try:
            if not self.registry.has_tool(tool_name):
                return ToolCallResult(success=False, error=f"Tool '{tool_name}' not found")

            tool_executor = self.registry.get_tool(tool_name)
            if tool_executor.is_async():
                result = self.run_async(tool_executor.execute, arguments)
            else:
                result = tool_executor.execute(arguments)
            return ToolCallResult(success=True, result=result)

        except ToolExecutionError as e:
            return ToolCallResult(success=False, error=str(e))
        except Exception as e:
            logging.error(f'Unexpected error executing tool {tool_name}: {e}')
            return ToolCallResult(success=False, error=f'Unexpected error: {e}')


def format_tools_for_prompt(tools: list[ToolSchema]) -> str:
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


def _convert_pydantic_arguments(func: typing.Callable, arguments: dict[str, typing.Any]) -> dict[str, typing.Any]:
    signature = inspect.signature(func)
    converted_args = {}

    for param_name, param in signature.parameters.items():
        if param_name in arguments:
            # check if this parameter is a Pydantic BaseModel
            try:
                if inspect.isclass(param.annotation) and issubclass(param.annotation, pydantic.BaseModel):
                    # convert dict to Pydantic model
                    converted_args[param_name] = param.annotation(**arguments[param_name])
                else:
                    converted_args[param_name] = arguments[param_name]
            except (TypeError, AttributeError):
                # if conversion fails or not a Pydantic model, use original value
                converted_args[param_name] = arguments[param_name]

    return converted_args


async def register_mcp_tools(tool_registry: 'ToolRegistry', mcp_server_manager: 'MCPServerManager') -> 'ToolRegistry':
    if not mcp_server_manager:
        logging.warning('No MCP server manager provided, skipping MCP tool registration')
        return tool_registry

    servers = mcp_server_manager.list_servers()
    for server_name in servers:
        server = mcp_server_manager.get_server(server_name)
        tool_registry = await _register_server_tools(tool_registry, server)

    return tool_registry


async def _register_server_tools(tool_registry: 'ToolRegistry', server: MCPServer) -> ToolRegistry:
    transport = server.get_transport()
    if not transport:
        logging.warning(f'No transport available for server {server.config.name}')
        return

    try:
        client = MCPClient(server)
        tools_list = await client.list_tools()

        for tool_info in tools_list:
            print(tool_info)
            tool_name = tool_info.get('name', 'unknown')
            description = tool_info.get('description', 'MCP tool')
            input_schema = tool_info.get('inputSchema', {})

            mcp_executor = create_mcp_tool_executor(
                tool_name=tool_name,
                description=description,
                input_schema=input_schema,
                server=server,
            )

            tool_registry.register(mcp_executor)
            logging.info(f'Registered MCP tool: {tool_name} from server {server.config.name}')

    except Exception as e:
        raise RuntimeError(f'Failed to register tools from server {server.config.name}: {e}')
    return tool_registry


def create_mcp_tool_executor(
    tool_name: str, description: str, input_schema: dict, server: MCPServer
) -> 'MCPToolExecutor':
    parameters = []

    if input_schema.get('type') == 'object':
        properties = input_schema.get('properties', {})
        required_fields = input_schema.get('required', [])

        for param_name, param_info in properties.items():
            parameters.append(
                ToolParameter(
                    name=param_name,
                    type_hint=param_info.get('type', 'str'),
                    required=param_name in required_fields,
                    description=param_info.get('description', ''),
                )
            )

    schema = ToolSchema(tool_name, description, parameters, ToolType.MCP_TOOL)
    return MCPToolExecutor(tool_name, server, schema)
