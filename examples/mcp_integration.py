import argparse
import asyncio
import contextlib
import dataclasses
import json
import logging
import os
import pathlib
import typing

import mcp
from mcp.client.stdio import stdio_client

from owl import tool


@dataclasses.dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None


class MCPServer:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self.session: mcp.ClientSession | None = None
        self.exit_stack: contextlib.AsyncExitStack = contextlib.AsyncExitStack()
        self._tools: list[tool.ToolSchema] = []
        self._cleanup_lock: asyncio.Lock = asyncio.Lock()

    async def initialize(self) -> None:
        server_params = mcp.StdioServerParameters(
            command=self.config.command, args=self.config.args, env={**os.environ, **(self.config.env or {})}
        )

        try:
            stdio_transport = await self.exit_stack.enter_async_context(stdio_client(server_params))
            read, write = stdio_transport
            self.session = await self.exit_stack.enter_async_context(mcp.ClientSession(read, write))
            await self.session.initialize()
            await self._load_tools()
            logging.info(f"Initialized MCP server '{self.config.name}' with {len(self._tools)} tools")
        except Exception as e:
            logging.error(f'Error initializing MCP server {self.config.name}: {e}')
            await self.cleanup()
            raise

    async def _load_tools(self) -> None:
        if not self.session:
            return

        tools_response = await self.session.list_tools()
        self._tools = []

        for item in tools_response:
            if isinstance(item, tuple) and item[0] == 'tools':
                for mcp_tool in item[1]:
                    schema = self._convert_mcp_tool_to_schema(mcp_tool)
                    self._tools.append(schema)

    def _convert_mcp_tool_to_schema(self, mcp_tool) -> tool.ToolSchema:
        parameters = []

        if hasattr(mcp_tool, 'inputSchema') and '$defs' in mcp_tool.inputSchema:
            key = list(mcp_tool.inputSchema['$defs'].keys())[0]
            properties = mcp_tool.inputSchema['$defs'][key].get('properties', {})
            required_fields = mcp_tool.inputSchema['$defs'][key].get('required', [])

            for param_name, param_info in properties.items():
                parameters.append(
                    tool.ToolParameter(
                        name=param_name,
                        type_hint=param_info.get('type', 'Any'),
                        required=param_name in required_fields,
                        description=param_info.get('description', ''),
                    )
                )

        return tool.ToolSchema(
            name=mcp_tool.name,
            description=mcp_tool.description,
            parameters=parameters,
            tool_type=tool.ToolType.MCP_TOOL,
            metadata={'server_name': self.config.name},
        )

    async def execute_tool(self, tool_name: str, arguments: dict[str, typing.Any]) -> typing.Any:
        if not self.session:
            raise RuntimeError(f'MCP server {self.config.name} not initialized')

        try:
            result = await self.session.call_tool(tool_name, {'request': arguments})
            return result
        except Exception as e:
            raise tool.ToolExecutionError(f'MCP tool execution failed: {e}')

    def get_tools(self) -> list[tool.ToolSchema]:
        return self._tools.copy()

    async def cleanup(self) -> None:
        async with self._cleanup_lock:
            try:
                if not self.session:
                    return
                self.session = None
                await self.exit_stack.aclose()
                self.stdio_context = None
                print(f"cleaned up {self.config.name}")
            except Exception as e:
                logging.error(f'Error during cleanup of server {self.config.name}: {e}')
                if not isinstance(e, RuntimeError) and "cancel scope in a different task" in str(e):
                    raise
            except Exception as e:
                logging.error(f'Error during cleanup of server {self.config.name}: {e}')


class MCPToolExecutor(tool.ToolExecutor):
    def __init__(self, server: MCPServer, schema: tool.ToolSchema):
        self.server = server
        self.schema = schema

    async def execute_async(self, arguments: dict[str, typing.Any]) -> typing.Any:
        return await self.server.execute_tool(self.schema.name, arguments)

    def execute(self, arguments: dict[str, typing.Any]) -> typing.Any:
        return asyncio.run(self.execute_async(arguments))

    def get_schema(self) -> tool.ToolSchema:
        return self.schema


async def setup_mcp_tools(
    config_path: str | pathlib.Path,
) -> tool.ToolRegistry:
    with open(config_path, 'r') as f:
        config_data = json.load(f)

    registry = tool.ToolRegistry()

    for server_name, server_config in config_data.get('mcpServers', {}).items():
        mcp_config = MCPServerConfig(
            name=server_name,
            command=server_config['command'],
            args=server_config['args'],
            env=server_config.get('env'),
        )

        server = MCPServer(mcp_config)
        try:
            await server.initialize()
        except Exception as ex:
            logging.error(f'Failed to initialize server {server_name}: {ex}')
            await server.cleanup()
            return

        for tool_schema in server.get_tools():
            executor = MCPToolExecutor(server, tool_schema)
            registry.register(executor)

        logging.info(f"Added MCP server '{mcp_config.name}' with {len(server.get_tools())} tools")

    return registry


async def run(args: argparse.Namespace):
    tool_registry = None
    try:
        tool_registry = await setup_mcp_tools(args.config)
        # tool_caller = tool.UniversalToolCaller(registry)
        tools = tool_registry.list_tools()
        print(f'\nFound {len(tools)} tools:\n')
        for tool_schema in tools:
            server_info = f' (MCP: {tool_schema.metadata["server_name"]})'
            print(f'- {tool_schema.name}: {tool_schema.description}{server_info}')

        print('MCP tools setup completed!')
        print(f'Loaded {len(tool_registry.list_tools())} tools from MCP servers')
        print('Servers are running. Press Ctrl+C to shutdown...')
        while True:
            await asyncio.sleep(1)

    except KeyboardInterrupt as ex:
        logging.error(f'Keyboard interrupt received: {ex}')
    except Exception as ex:
        logging.error(f'Error: {ex}')
    finally:
        if tool_registry and tool_registry._tools:
            print('Cleaning up servers...')
            # temp hack
            for tool in reversed(tool_registry._tools):
                try:
                    if hasattr(tool, 'server'):
                        await tool.server.cleanup()
                except Exception as e:
                    logging.error(f'Error cleaning up server {tool.server}: {e}')
            print('Cleanup completed.')
            current = asyncio.current_task()
            pending = [t for t in asyncio.all_tasks() if t is not current]
            for task in pending:
                task.cancel()
            await asyncio.gather(*pending, return_exceptions=True)


def main():
    parser = argparse.ArgumentParser(description='MCP Tools Setup')
    parser.add_argument('--config', '-c', required=True, help='Path to MCP server configuration JSON file')
    parser.add_argument('--verbose', '-v', action='store_true', help='Enable verbose logging')

    args = parser.parse_args()

    level = logging.DEBUG if args.verbose else logging.INFO
    logging.basicConfig(level=level, format='%(asctime)s - %(levelname)s - %(message)s')
    asyncio.run(run(args))


if __name__ == '__main__':
    main()
