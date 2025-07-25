import abc
import asyncio
import enum
import subprocess
import sys
from typing import Optional


class TransportType(enum.Enum):
    INMEMORY = 'inmemory'
    STDIO = 'stdio'
    WEBSOCKET = 'websocket'
    TCP = 'tcp'
    UDP = 'udp'
    HTTP = 'http'
    DUMMY = 'dummy'


class Transport(abc.ABC):
    @abc.abstractmethod
    async def receive(self) -> str:
        pass

    @abc.abstractmethod
    async def send(self, message: str) -> None:
        pass

    @abc.abstractmethod
    async def close(self) -> None:
        pass


class StdioTransport(Transport):
    def __init__(self, process: subprocess.Popen = None, timeout: float = 10.0):
        self.process = process
        self._closed = False
        self._timeout = timeout

        if process:
            # Subprocess mode - communicate with external process
            self.stdin_stream = process.stdin
            self.stdout_stream = process.stdout
            self._is_subprocess = True
        else:
            # Host mode - communicate via our own stdio
            self.stdin_stream = sys.stdin
            self.stdout_stream = sys.stdout
            self._is_subprocess = False
            self._stdin_reader = None

    async def receive(self) -> str:
        if self._closed:
            raise RuntimeError('Transport is closed')

        if self._is_subprocess:
            # Subprocess mode - use executor for blocking reads with timeout
            if self.stdout_stream is None:
                raise RuntimeError('Process stdout is not available')

            loop = asyncio.get_event_loop()
            try:
                # Add timeout to prevent hanging
                line = await asyncio.wait_for(
                    loop.run_in_executor(None, self.stdout_stream.readline), timeout=self._timeout
                )
            except asyncio.TimeoutError:
                raise EOFError('Timeout waiting for response')

            if not line:
                raise EOFError('EOF reached')
            return line.strip()
        else:
            # Host mode - use asyncio streams
            await self._ensure_streams()
            line = await self._stdin_reader.readline()
            if not line:
                raise EOFError('EOF reached')
            return line.decode().strip()

    async def send(self, message: str) -> None:
        if self._closed:
            raise RuntimeError('Transport is closed')

        if self._is_subprocess:
            # Subprocess mode - use executor for blocking writes
            if self.stdin_stream is None:
                raise RuntimeError('Process stdin is not available')

            loop = asyncio.get_event_loop()
            await loop.run_in_executor(None, self._write_to_stdin, message)
        else:
            # Host mode - direct write to stdout
            self.stdout_stream.write(message + '\n')
            self.stdout_stream.flush()

    def _write_to_stdin(self, message: str) -> None:
        self.stdin_stream.write(message + '\n')
        self.stdin_stream.flush()

    async def _ensure_streams(self):
        if self._stdin_reader is None:
            self._stdin_reader = asyncio.StreamReader()
            protocol = asyncio.StreamReaderProtocol(self._stdin_reader)
            await asyncio.get_event_loop().connect_read_pipe(lambda: protocol, self.stdin_stream)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self._is_subprocess and self.stdin_stream:
                self.stdin_stream.close()


class InMemoryTransport(Transport):
    def __init__(self):
        self._incoming_queue = asyncio.Queue()
        self._outgoing_queue = asyncio.Queue()
        self._closed = False

    async def receive(self) -> str:
        if self._closed:
            raise RuntimeError('Transport is closed')

        message = await self._incoming_queue.get()
        if message is None:  # Sentinel for close
            raise EOFError('Transport closed')
        return message

    async def send(self, message: str) -> None:
        if self._closed:
            raise RuntimeError('Transport is closed')

        await self._outgoing_queue.put(message)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._incoming_queue.put(None)  # Sentinel to unblock receive

    async def inject_message(self, message: str):
        await self._incoming_queue.put(message)

    async def get_sent_message(self) -> str:
        return await self._outgoing_queue.get()

    def get_sent_message_nowait(self) -> Optional[str]:
        try:
            return self._outgoing_queue.get_nowait()
        except asyncio.QueueEmpty:
            return None


class HttpTransport(Transport):
    def __init__(self, request_handler, response_sender):
        self.request_handler = request_handler
        self.response_sender = response_sender
        self._request_queue = asyncio.Queue()
        self._response_queue = asyncio.Queue()
        self._closed = False

    async def receive(self) -> str:
        if self._closed:
            raise RuntimeError('Transport is closed')

        message = await self._request_queue.get()
        if message is None:
            raise EOFError('Transport closed')
        return message

    async def send(self, message: str) -> None:
        if self._closed:
            raise RuntimeError('Transport is closed')

        await self._response_queue.put(message)

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            await self._request_queue.put(None)

    async def handle_http_request(self, request_body: str) -> str:
        await self._request_queue.put(request_body)
        return await self._response_queue.get()


class TcpTransport(Transport):
    def __init__(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter):
        self.reader = reader
        self.writer = writer
        self._closed = False

    async def receive(self) -> str:
        if self._closed:
            raise RuntimeError('Transport is closed')

        try:
            line = await self.reader.readline()
            if not line:
                raise EOFError('Connection closed')
            return line.decode().strip()
        except Exception:
            self._closed = True
            raise

    async def send(self, message: str) -> None:
        if self._closed:
            raise RuntimeError('Transport is closed')

        try:
            self.writer.write((message + '\n').encode())
            await self.writer.drain()
        except Exception:
            self._closed = True
            raise

    async def close(self) -> None:
        if not self._closed:
            self._closed = True
            if self.writer:
                self.writer.close()
                await self.writer.wait_closed()


class DummyTransport(Transport):
    def __init__(self):
        self._closed = False

    async def receive(self) -> str:
        if self._closed:
            raise RuntimeError('Transport is closed')
        await asyncio.sleep(0.1)
        return '{"jsonrpc": "2.0", "result": "dummy_response", "id": 1}'

    async def send(self, message: str) -> None:
        if self._closed:
            raise RuntimeError('Transport is closed')
        pass

    async def close(self) -> None:
        self._closed = True


# kwargs?
def create_transport(transport_type: TransportType) -> Transport:
    if transport_type == TransportType.STDIO:
        return StdioTransport()
    elif transport_type == TransportType.INMEMORY:
        return InMemoryTransport()
    elif transport_type == TransportType.HTTP:
        return HttpTransport(None, None)  # Placeholder for HTTP
    elif transport_type == TransportType.DUMMY:
        return DummyTransport()
    else:
        raise ValueError(f'Unsupported transport type: {transport_type}')
