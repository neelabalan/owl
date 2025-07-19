import argparse
import asyncio
import json
import logging
import pathlib
import subprocess
import threading
from dataclasses import dataclass

from owl.transports import TransportType
from owl.transports import create_transport


@dataclass
class MCPServerConfig:
    name: str
    command: str
    args: list[str]
    env: dict[str, str] | None = None
    transport_type: TransportType = TransportType.STDIO
    process_timeout: float = 5.0
    startup_timeout: float = 5.0


class MCPServer:
    def __init__(self, config: MCPServerConfig):
        self.config = config
        self._process = None
        self.transport = None
        self._lock = threading.Lock()

    def start(self) -> subprocess.Popen:
        with self._lock:
            if self._process is not None:
                return self._process

            # Spawn the MCP server process
            self._process = subprocess.Popen(
                [self.config.command] + self.config.args,
                stdin=subprocess.PIPE,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                env=self.config.env,
                text=True,
                bufsize=0,
            )

            logging.info(f"Spawned MCP server '{self.config.name}' with PID {self._process.pid}")

            if not self._wait_for_process_ready():
                self._terminate_process()
                raise RuntimeError(f"MCP server '{self.config.name}' failed to start")

            if self.config.transport_type == TransportType.STDIO:
                from owl.transports import StdioTransport

                self.transport = StdioTransport(process=self._process)
            else:
                # for other transport types, use the factory
                self.transport = create_transport(self.config.transport_type)

            logging.info(f"Started MCP server '{self.config.name}' with {self.config.transport_type.value} transport")
            return self._process

    def _wait_for_process_ready(self) -> bool:
        import time

        if self._process is None:
            return False

        # for STDIO, process is ready immediately after spawn
        if self.config.transport_type == TransportType.STDIO:
            return self._process.poll() is None

        # for network transports, we might need to wait for port binding
        # for now, just check if process is still running
        start_time = time.time()
        while time.time() - start_time < self.config.startup_timeout:
            if self._process.poll() is not None:  # Process died
                return False
            time.sleep(0.1)

        return True

    def _terminate_process(self) -> None:
        if self._process is None:
            return

        self._process.terminate()
        try:
            self._process.wait(timeout=self.config.process_timeout)
        except subprocess.TimeoutExpired:
            logging.warning(f"Process '{self.config.name}' did not terminate gracefully, forcing kill")
            self._process.kill()
            self._process.wait()

        self._process = None

    async def stop(self) -> None:
        with self._lock:
            if self.transport is not None:
                await self.transport.close()
                self.transport = None

            self._terminate_process()
            logging.info(f"Stopped MCP server '{self.config.name}'")

    def is_running(self) -> bool:
        with self._lock:
            return self.transport is not None and self._process is not None and self._process.poll() is None

    def get_transport(self):
        return self.transport

    def get_process(self) -> subprocess.Popen | None:
        with self._lock:
            return self._process


class MCPServerManager:
    def __init__(self, config_path: str | pathlib.Path):
        self.config_path = pathlib.Path(config_path)
        self.servers: dict[str, MCPServer] = {}

    def start(self) -> None:
        with open(self.config_path, 'r') as f:
            config_data = json.load(f)

        for server_name, server_config in config_data.get('mcpServers', {}).items():
            transport_type_str = server_config.get('transport', 'stdio')
            transport_type = TransportType(transport_type_str)

            mcp_config = MCPServerConfig(
                name=server_name,
                command=server_config['command'],
                args=server_config['args'],
                env=server_config.get('env'),
                transport_type=transport_type,
            )

            server = MCPServer(mcp_config)
            self.servers[server_name] = server
            server.start()  # subprocess.Popen() is non-blocking

        logging.info(f'Started {len(self.servers)} MCP servers')

    async def stop(self) -> None:
        tasks = []

        for server in self.servers.values():
            task = asyncio.create_task(server.stop())
            tasks.append(task)

        await asyncio.gather(*tasks, return_exceptions=True)
        logging.info('All MCP servers stopped')

    def get_server(self, name: str) -> MCPServer | None:
        return self.servers.get(name)

    def list_servers(self) -> list[str]:
        return list(self.servers.keys())

    def get_transport(self, server_name: str):
        server = self.get_server(server_name)
        return server.get_transport() if server else None


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='MCP Server Manager')
    parser.add_argument('--config', '-c', required=True, help='Path to MCP server configuration JSON file')
    parser.add_argument('--debug', '-d', action='store_true', help='Enable debug logging')

    args = parser.parse_args()

    level = logging.DEBUG if args.debug else logging.INFO
    logging.basicConfig(
        level=level, format='%(asctime)s - %(levelname)s - %(message)s', handlers=[logging.StreamHandler()]
    )

    server_manager = MCPServerManager(args.config)

    try:
        server_manager.start()
        logging.info('MCP servers started successfully')
        logging.info(f'Running servers: {server_manager.list_servers()}')

        import signal

        signal.pause()

    except KeyboardInterrupt:
        logging.info('Shutdown requested')
    except Exception as e:
        logging.error(f'Error: {e}')
    finally:
        import asyncio

        asyncio.run(server_manager.stop())
        logging.info('All servers stopped')
