import abc
import json
import logging
import typing
import uuid
from enum import Enum

import pydantic

from owl.tool import ToolRegistry
from owl.transports import Transport


class JsonRpcError(Exception):
    def __init__(self, code: int, message: str, data: typing.Any = None):
        self.code = code
        self.message = message
        self.data = data
        super().__init__(f'JSON-RPC Error {code}: {message}')


class ErrorCode(Enum):
    PARSE_ERROR = -32700
    INVALID_REQUEST = -32600
    METHOD_NOT_FOUND = -32601
    INVALID_PARAMS = -32602
    INTERNAL_ERROR = -32603


class JsonRpcErrorModel(pydantic.BaseModel):
    code: int
    message: str
    data: typing.Any = None

    class Config:
        extra = 'allow'


class JsonRpcRequest(pydantic.BaseModel):
    method: str = pydantic.Field(..., min_length=1, description='The name of the method to be invoked')
    params: typing.Any = pydantic.Field(default=None, description='Parameters for the method')
    id: typing.Union[str, int, None] = pydantic.Field(default=None, description='Request identifier')
    jsonrpc: str = pydantic.Field(default='2.0', pattern=r'^2\.0$', description='JSON-RPC version')

    class Config:
        extra = 'forbid'

    @classmethod
    def from_dict(cls, data: dict) -> 'JsonRpcRequest':
        return cls(**data)

    def is_notification(self) -> bool:
        return self.id is None


class JsonRpcResponse(pydantic.BaseModel):
    result: typing.Any = None
    error: typing.Optional[JsonRpcErrorModel] = None
    id: typing.Union[str, int, None] = None
    jsonrpc: str = pydantic.Field(default='2.0')

    class Config:
        extra = pydantic.Extra.forbid

    @classmethod
    def create_success(cls, result: typing.Any, request_id: typing.Union[str, int, None]) -> 'JsonRpcResponse':
        return cls(result=result, id=request_id)

    @classmethod
    def create_error(
        cls, code: int, message: str, request_id: typing.Union[str, int, None], data: typing.Any = None
    ) -> 'JsonRpcResponse':
        error_obj = JsonRpcErrorModel(code=code, message=message, data=data)
        return cls(error=error_obj, id=request_id)

    def to_dict(self) -> dict:
        return self.model_dump(exclude_none=True)


class Middleware(abc.ABC):
    @abc.abstractmethod
    async def process_request(self, request: JsonRpcRequest) -> JsonRpcRequest:
        pass

    @abc.abstractmethod
    async def process_response(self, response: JsonRpcResponse, request: JsonRpcRequest) -> JsonRpcResponse:
        pass


# not sure if this'd be required
class Protocol(abc.ABC):
    @abc.abstractmethod
    def parse_request(self, raw_message: str) -> JsonRpcRequest:
        pass

    @abc.abstractmethod
    def format_response(self, response: JsonRpcResponse) -> str:
        pass

    @abc.abstractmethod
    def format_request(self, request: JsonRpcRequest) -> str:
        pass


class StandardJsonRpcProtocol(Protocol):
    def parse_request(self, raw_message: str) -> JsonRpcRequest:
        try:
            data = json.loads(raw_message)
            return JsonRpcRequest.from_dict(data)
        except json.JSONDecodeError as e:
            raise JsonRpcError(ErrorCode.PARSE_ERROR.value, f'Parse error: {e}')
        except pydantic.ValidationError as e:
            raise JsonRpcError(ErrorCode.INVALID_REQUEST.value, f'Invalid request: {e}')
        except KeyError as e:
            raise JsonRpcError(ErrorCode.INVALID_REQUEST.value, f'Invalid request: missing {e}')

    def format_response(self, response: JsonRpcResponse) -> str:
        return response.model_dump_json(exclude_none=True)

    def format_request(self, request: JsonRpcRequest) -> str:
        return request.model_dump_json(exclude_none=True)


class JsonRpcEngine:
    def __init__(
        self,
        transport: Transport,
        tool_registry: ToolRegistry,
        protocol: Protocol = None,
        middleware: list[Middleware] = None,
    ):
        self.transport = transport
        self.tool_registry = tool_registry
        self.protocol = protocol or StandardJsonRpcProtocol()
        self.middleware = middleware or []
        self._running = False

    def _convert_params_to_arguments(self, params: typing.Any) -> dict:
        if params is None:
            return {}
        elif isinstance(params, dict):
            return params
        elif isinstance(params, list):
            # handle positional parameters by converting to dict with numeric keys
            return {str(i): param for i, param in enumerate(params)}
        else:
            return {'value': params}

    def _execute_method_sync(self, method_name: str, params: typing.Any) -> typing.Any:
        if not self.tool_registry.has_tool(method_name):
            raise JsonRpcError(ErrorCode.METHOD_NOT_FOUND.value, f"Method '{method_name}' not found")

        try:
            tool_executor = self.tool_registry.get_tool(method_name)
            arguments = self._convert_params_to_arguments(params)
            return tool_executor.execute(arguments)
        except JsonRpcError:
            raise
        except Exception as e:
            logging.exception(f"Error executing method '{method_name}': {e}")
            raise JsonRpcError(ErrorCode.INTERNAL_ERROR.value, 'Internal server error')

    async def _execute_method_async(self, method_name: str, params: typing.Any) -> typing.Any:
        if not self.tool_registry.has_tool(method_name):
            raise JsonRpcError(ErrorCode.METHOD_NOT_FOUND.value, f"Method '{method_name}' not found")

        try:
            tool_executor = self.tool_registry.get_tool(method_name)
            arguments = self._convert_params_to_arguments(params)
            return await tool_executor.execute(arguments)
        except JsonRpcError:
            raise
        except Exception as e:
            logging.exception(f"Error executing method '{method_name}': {e}")
            raise JsonRpcError(ErrorCode.INTERNAL_ERROR.value, 'Internal server error')

    async def handle_request(self, raw_message: str) -> str:
        try:
            request = self.protocol.parse_request(raw_message)

            for middleware in self.middleware:
                request = await middleware.process_request(request)
            if not self.tool_registry.has_tool(request.method):
                raise JsonRpcError(ErrorCode.METHOD_NOT_FOUND.value, f"Method '{request.method}' not found")

            if self.tool_registry.is_tool_async(request.method):
                result = await self._execute_method_async(request.method, request.params)
            else:
                result = self._execute_method_sync(request.method, request.params)

            if request.is_notification():
                return None

            response = JsonRpcResponse.create_success(result, request.id)
            for middleware in reversed(self.middleware):
                response = await middleware.process_response(response, request)

            return self.protocol.format_response(response)

        except JsonRpcError as e:
            if hasattr(request, 'id'):
                response = JsonRpcResponse.create_error(e.code, e.message, request.id, e.data)
            else:
                response = JsonRpcResponse.create_error(e.code, e.message, None, e.data)
            return self.protocol.format_response(response)
        except Exception as e:
            logging.exception(f'Unexpected error handling request: {e}')
            response = JsonRpcResponse.create_error(
                ErrorCode.INTERNAL_ERROR.value,
                'Internal server error',
                getattr(request, 'id', None) if 'request' in locals() else None,
            )
            return self.protocol.format_response(response)

    async def start(self):
        self._running = True
        try:
            while self._running:
                try:
                    raw_message = await self.transport.receive()
                    response = await self.handle_request(raw_message)
                    if response:
                        await self.transport.send(response)
                except Exception as e:
                    logging.exception(f'Error processing message: {e}')
                    continue
        except KeyboardInterrupt:
            logging.info('Received interrupt signal, shutting down gracefully...')
        except Exception as e:
            logging.exception(f'Fatal error in main loop: {e}')
        finally:
            await self.stop()

    async def stop(self):
        self._running = False
        try:
            await self.transport.close()
        except Exception as e:
            logging.exception(f'Error during transport cleanup: {e}')

    async def call_method(self, method: str, params: typing.Any = None) -> typing.Any:
        request_id = str(uuid.uuid4())
        request = JsonRpcRequest(method=method, params=params, id=request_id)
        request_str = self.protocol.format_request(request)

        await self.transport.send(request_str)
        response_str = await self.transport.receive()

        try:
            response_data = json.loads(response_str)
            if 'error' in response_data:
                error = response_data['error']
                raise JsonRpcError(error['code'], error['message'], error.get('data'))
            return response_data.get('result')
        except json.JSONDecodeError as e:
            raise JsonRpcError(ErrorCode.PARSE_ERROR.value, f'Invalid response: {e}')


class LoggingMiddleware(Middleware):
    async def process_request(self, request: JsonRpcRequest) -> JsonRpcRequest:
        logging.info(f'Incoming request: {request.method} with params: {request.params}')
        return request

    async def process_response(self, response: JsonRpcResponse, request: JsonRpcRequest) -> JsonRpcResponse:
        if response.error:
            logging.error(f'Request {request.method} failed: {response.error.model_dump()}')
        else:
            logging.info(f'Request {request.method} completed successfully')
        return response
