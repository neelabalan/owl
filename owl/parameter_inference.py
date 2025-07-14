import abc
import dataclasses
import inspect
import typing


@dataclasses.dataclass
class ParameterInfo:
    name: str
    type_hint: str
    required: bool = True
    description: str = ''
    default: typing.Any = None
    constraints: dict[str, typing.Any] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}


class ParameterParser(abc.ABC):
    """Abstract base class for parameter parsers"""

    @abc.abstractmethod
    def can_parse(self, annotation: typing.Any) -> bool:
        """Check if this parser can handle the given annotation"""
        pass

    @abc.abstractmethod
    def parse(self, param_name: str, param: inspect.Parameter) -> ParameterInfo:
        """Parse the parameter and return ParameterInfo"""
        pass


class AnnotatedTypeParser(ParameterParser):
    def can_parse(self, annotation: typing.Any) -> bool:
        return typing.get_origin(annotation) is typing.Annotated

    def parse(self, param_name: str, param: inspect.Parameter) -> ParameterInfo:
        annotation = param.annotation

        # extract the actual type and metadata using get_args
        args = typing.get_args(annotation)
        actual_type = args[0]
        metadata = args[1:] if len(args) > 1 else ()

        description = ''
        constraints = {}

        # process metadata
        for meta in metadata:
            if isinstance(meta, str):
                description = meta
            elif isinstance(meta, dict):
                constraints.update(meta)

        type_hint = self._format_type_hint(actual_type)
        required = param.default == inspect.Parameter.empty
        default = param.default if param.default != inspect.Parameter.empty else None

        return ParameterInfo(
            name=param_name,
            type_hint=type_hint,
            required=required,
            description=description,
            default=default,
            constraints=constraints,
        )

    def _format_type_hint(self, type_obj: typing.Any) -> str:
        if hasattr(type_obj, '__name__'):
            return type_obj.__name__
        else:
            # handle complex types like list[str], dict[str, int], etc.
            type_str = str(type_obj)
            # clean up the representation
            type_str = type_str.replace('typing.', '')
            type_str = type_str.replace("<class '", '').replace("'>", '')
            return type_str


class StandardTypeParser(ParameterParser):
    def can_parse(self, annotation: typing.Any) -> bool:
        return annotation != inspect.Parameter.empty

    def parse(self, param_name: str, param: inspect.Parameter) -> ParameterInfo:
        type_hint = self._format_type_hint(param.annotation)
        required = param.default == inspect.Parameter.empty
        default = param.default if param.default != inspect.Parameter.empty else None

        return ParameterInfo(name=param_name, type_hint=type_hint, required=required, description='', default=default)

    def _format_type_hint(self, type_obj: typing.Any) -> str:
        if hasattr(type_obj, '__name__'):
            return type_obj.__name__
        else:
            # handle complex types like list[str], dict[str, int], Union, Optional, etc.
            type_str = str(type_obj)

            # clean up common typing representations
            replacements = {
                'typing.': '',
                "<class '": '',
                "'>": '',
                'builtins.': '',
            }

            for old, new in replacements.items():
                type_str = type_str.replace(old, new)

            return type_str


class DefaultTypeParser(ParameterParser):
    def can_parse(self, annotation: typing.Any) -> bool:
        return True  # Always can parse as fallback

    def parse(self, param_name: str, param: inspect.Parameter) -> ParameterInfo:
        required = param.default == inspect.Parameter.empty
        default = param.default if param.default != inspect.Parameter.empty else None

        return ParameterInfo(name=param_name, type_hint='Any', required=required, description='', default=default)


class ParameterInferenceEngine:
    def __init__(self, parsers: list[ParameterParser] = None):
        # order matters - more specific parsers first
        if not parsers:
            self.parsers: list[ParameterParser] = [AnnotatedTypeParser(), StandardTypeParser(), DefaultTypeParser()]

    def infer_parameters(self, func: typing.Callable) -> list[ParameterInfo]:
        signature = inspect.signature(func)
        parameters = []

        for param_name, param in signature.parameters.items():
            # find the first parser that can handle this parameter
            for parser in self.parsers:
                if parser.can_parse(param.annotation):
                    param_info = parser.parse(param_name, param)
                    parameters.append(param_info)
                    break

        return parameters

    def format_constraints(self, constraints: dict[str, typing.Any]) -> str:
        if not constraints:
            return ''

        def format_examples(examples):
            quoted_examples = [f"'{ex}'" for ex in examples]
            return f'examples: {", ".join(quoted_examples)}'

        formatters = {
            'min_length': lambda v: f'min length: {v}',
            'max_length': lambda v: f'max length: {v}',
            'pattern': lambda v: f'pattern: {v}',
            'examples': format_examples,
            'min_value': lambda v: f'min: {v}',
            'max_value': lambda v: f'max: {v}',
        }

        parts = [formatters[key](value) for key, value in constraints.items() if key in formatters]
        return f' ({", ".join(parts)})' if parts else ''
