import abc
import dataclasses
import datetime
import enum
import typing
import uuid


class Role(enum.Enum):
    user = 'user'
    agent = 'agent'
    tool = 'tool'


class TokenLimitExceeded(Exception):
    def __init__(self, message='Token limit exceeded'):
        super().__init__(message)


class AgentError(Exception):
    pass


@dataclasses.dataclass(frozen=True)
class Tool:
    name: str
    description: str
    parameters: dict[str, typing.Any]
    function: typing.Callable

    def call(self, **kwargs) -> typing.Any:
        """Execute the tool function"""
        return self.function(**kwargs)


@dataclasses.dataclass(frozen=True)
class Message:
    author: str
    content: str = ''
    timestamp: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)
    tool_calls: list[dict[str, typing.Any]] = dataclasses.field(default_factory=list)
    tool_call_id: str | None = None


# Observer Pattern for Monitoring
class Observer(abc.ABC):
    @abc.abstractmethod
    def update(self, run_id: str, message: Message):
        pass


# Run lifecycle - https://platform.openai.com/docs/assistants/deep-dive#run-lifecycle
class ThreadState(enum.Enum):
    QUEUED = 'queued'
    IN_PROGRESS = 'in_progress'
    COMPLETED = 'completed'
    REQUIRES_ACTION = 'requires_action'
    EXPIRED = 'expired'
    CANCELLING = 'cancelling'
    CANCELLED = 'cancelled'
    FAILED = 'failed'
    INCOMPLETE = 'incomplete'


@dataclasses.dataclass(slots=True)
class Thread:
    id: str = dataclasses.field(default_factory=lambda: str(uuid.uuid4()))
    messages: list[Message] = dataclasses.field(default_factory=list)
    metadata: dict[str, typing.Any] = dataclasses.field(default_factory=dict)
    created_at: datetime.datetime = dataclasses.field(default_factory=datetime.datetime.now)

    def add_message(self, message: Message) -> None:
        self.messages.append(message)

    # this means current message
    def get_last_message(self) -> Message | None:
        pass


class AgentManager:
    def list_agents(self): ...

    def create_agent(self): ...

    def get_agent(self): ...


@dataclasses.dataclass(frozen=True)
class Human:
    name: str
    role: Role


@dataclasses.dataclass
class Agent:
    instruction: str
    name: str
    role: Role
    description: str = ''
    tools: list[Tool] = dataclasses.field(default_factory=list)
    model: str = 'gpt-4o'
    temperature: float = 0.0
    seed: int = 13
    observers: dict[str, Observer] = dataclasses.field(default_factory=dict)

    def run(self, prompt: str) -> str:
        pass

    class Config:
        allow_mutation = False


Entity = Human | Agent


@dataclasses.dataclass
class Conversation:
    participants: list[Entity]
    thread: Thread = dataclasses.field(default_factory=Thread)

    def engage(self, message: Message):
        pass
