import argparse
import asyncio
import json
import sys
from typing import Any

from owl.jsonrpc import JsonRpcNotification
from owl.jsonrpc import JsonRpcRequest
from owl.mcp_manager import MCPServerManager


async def send_mcp_request(transport, request: JsonRpcRequest) -> dict[str, Any] | None:
    request_json = request.model_dump_json(exclude_none=True)
    print(f'Sending: {request_json}', file=sys.stderr)
    await transport.send(request_json)

    print('Waiting for response...', file=sys.stderr)
    try:
        response_line = await transport.receive()
        print(f'Received: {response_line}', file=sys.stderr)

        if response_line:
            return json.loads(response_line)
    except EOFError as e:
        print(f'Transport error: {e}', file=sys.stderr)
    except Exception as e:
        print(f'Error receiving response: {e}', file=sys.stderr)

    return None


async def list_tools_for_server(server_name: str, transport) -> None:
    print(f'\nQuerying server: {server_name}', file=sys.stderr)

    try:
        init_request = JsonRpcRequest(
            jsonrpc='2.0',
            id=1,
            method='initialize',
            params={
                'protocolVersion': '2024-11-05',
                'capabilities': {},
                'clientInfo': {'name': 'owl-mcp-client', 'version': '1.0.0'},
            },
        )

        init_response = await send_mcp_request(transport, init_request)
        if not init_response or init_response.get('error'):
            error_msg = init_response.get('error') if init_response else 'No response'
            print(f'Failed to initialize {server_name}: {error_msg}', file=sys.stderr)
            return

        notification = JsonRpcNotification(jsonrpc='2.0', method='notifications/initialized')
        print(f'Sending initialized notification: {notification.model_dump_json(exclude_none=True)}', file=sys.stderr)
        await transport.send(notification.model_dump_json(exclude_none=True))

        tools_request = JsonRpcRequest(jsonrpc='2.0', id=2, method='tools/list', params={})
        tools_response = await send_mcp_request(transport, tools_request)

        if tools_response and tools_response.get('result'):
            tools = tools_response['result'].get('tools', [])
            print(f'{server_name}: Found {len(tools)} tools', file=sys.stderr)

            for tool in tools:
                name = tool.get('name', 'unknown')
                description = tool.get('description', 'No description')
                print(f'  - {name}: {description}', file=sys.stderr)

                if 'inputSchema' in tool:
                    schema = tool['inputSchema']
                    properties = schema.get('properties', {})
                    if properties:
                        print(f'    Parameters: {list(properties.keys())}', file=sys.stderr)
        else:
            print(f'{server_name}: No tools found or error occurred', file=sys.stderr)

    except Exception as e:
        print(f'Error communicating with {server_name}: {e}', file=sys.stderr)


async def main() -> None:
    parser = argparse.ArgumentParser(description='List tools from MCP servers')
    parser.add_argument('--config', '-c', required=True, help='Path to MCP server configuration JSON file')

    args = parser.parse_args()

    server_manager = MCPServerManager(args.config)

    try:
        print('Starting MCP servers...', file=sys.stderr)
        server_manager.start()

        servers = server_manager.list_servers()
        print(f'Started {len(servers)} servers: {", ".join(servers)}', file=sys.stderr)

        for server_name in servers:
            transport = server_manager.get_transport(server_name)
            if transport:
                try:
                    await list_tools_for_server(server_name, transport)
                except asyncio.TimeoutError:
                    print(f'Timeout querying server {server_name}', file=sys.stderr)
                except asyncio.CancelledError:
                    print(f'Cancelled querying server {server_name}', file=sys.stderr)
                    break
            else:
                print(f'No transport available for {server_name}', file=sys.stderr)

    except KeyboardInterrupt:
        print('\nKeyboard interrupt received, stopping...', file=sys.stderr)
    finally:
        print('Stopping servers...', file=sys.stderr)
        try:
            await server_manager.stop()
        except asyncio.TimeoutError:
            print('Timeout stopping servers', file=sys.stderr)


if __name__ == '__main__':
    asyncio.run(main())
