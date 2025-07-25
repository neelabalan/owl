import abc
import dataclasses
import inspect
import typing

import pydantic


@dataclasses.dataclass
class ParameterInfo:
    name: str
    type_hint: str
    required: bool = True
    description: str = ''
    default: typing.Any = None
    constraints: dict[str, typing.Any] = None
    nested_fields: list['ParameterInfo'] = None

    def __post_init__(self):
        if self.constraints is None:
            self.constraints = {}
        if self.nested_fields is None:
            self.nested_fields = []


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


class PydanticTypeParser(ParameterParser):
    def can_parse(self, annotation: typing.Any) -> bool:
        try:
            return inspect.isclass(annotation) and issubclass(annotation, pydantic.BaseModel)
        except (TypeError, AttributeError):
            return False

    def parse(self, param_name: str, param: inspect.Parameter) -> ParameterInfo:
        model_class = param.annotation

        # extract information from the Pydantic model
        description = self._extract_model_description(model_class)
        constraints = self._extract_validation_constraints(model_class)
        nested_fields = self._extract_nested_fields(model_class)

        # use the model class name as the type hint
        type_hint = model_class.__name__

        required = param.default == inspect.Parameter.empty
        default = param.default if param.default != inspect.Parameter.empty else None

        return ParameterInfo(
            name=param_name,
            type_hint=type_hint,
            required=required,
            description=description,
            default=default,
            constraints=constraints,
            nested_fields=nested_fields,
        )

    def _extract_model_description(self, model_class) -> str:
        if hasattr(model_class, '__doc__') and model_class.__doc__:
            # use the first line of docstring as description
            return model_class.__doc__.strip().split('\n')[0]

        # try to get from model config
        if hasattr(model_class, 'model_config') and hasattr(model_class.model_config, 'description'):
            return model_class.model_config.description

        return f'{model_class.__name__} model'

    def _extract_validation_constraints(self, model_class) -> dict[str, typing.Any]:
        constraints = {}

        try:
            if hasattr(model_class, 'model_fields'):
                fields = model_class.model_fields

                for field_name, field_info in fields.items():
                    if hasattr(field_info, 'metadata'):
                        for constraint in field_info.metadata:
                            if hasattr(constraint, 'pattern'):
                                constraints['pattern'] = constraint.pattern
                            # add other constraint types as needed
        except Exception:
            pass

        return constraints

    def _extract_nested_fields(self, model_class) -> list[ParameterInfo]:
        nested_fields = []

        try:
            if hasattr(model_class, 'model_fields'):
                fields = model_class.model_fields

                for field_name, field_info in fields.items():
                    field_type = self._get_field_type(field_info)
                    description = getattr(field_info, 'description', '') or ''

                    # determine if field is required
                    is_required = True
                    default_value = None
                    if hasattr(field_info, 'default'):
                        if field_info.default is not None and str(field_info.default) != 'PydanticUndefined':
                            default_value = field_info.default
                            is_required = False

                    # extract validation constraints for this field
                    field_constraints = {}
                    if hasattr(field_info, 'metadata'):
                        for constraint in field_info.metadata:
                            if hasattr(constraint, 'pattern'):
                                field_constraints['pattern'] = constraint.pattern

                    nested_param = ParameterInfo(
                        name=field_name,
                        type_hint=field_type,
                        required=is_required,
                        description=description,
                        default=default_value,
                        constraints=field_constraints,
                    )

                    # handle deeply nested Pydantic models
                    if hasattr(field_info, 'annotation'):
                        try:
                            annotation = field_info.annotation
                            if inspect.isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
                                nested_param.nested_fields = self._extract_nested_fields(annotation)
                        except (TypeError, AttributeError):
                            pass

                    nested_fields.append(nested_param)
        except Exception:
            pass

        return nested_fields

    def _get_field_type(self, field_info) -> str:
        if hasattr(field_info, 'annotation'):
            annotation = field_info.annotation

            # handle nested Pydantic models
            try:
                if inspect.isclass(annotation) and issubclass(annotation, pydantic.BaseModel):
                    return f'{annotation.__name__} (nested object)'
            except (TypeError, AttributeError):
                pass

            # handle list types
            if hasattr(annotation, '__origin__') and annotation.__origin__ is list:
                args = getattr(annotation, '__args__', ())
                if args:
                    inner_type = self._format_simple_type(args[0])
                    return f'list[{inner_type}]'
                return 'list'

            # handle basic types
            if hasattr(annotation, '__name__'):
                return annotation.__name__
            else:
                # handle complex types
                return self._format_simple_type(annotation)

        return 'str'  # fallback

    def _format_simple_type(self, type_obj) -> str:
        """Format a simple type annotation"""
        if hasattr(type_obj, '__name__'):
            return type_obj.__name__

        type_str = str(type_obj)
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
            self.parsers: list[ParameterParser] = [
                AnnotatedTypeParser(),
                PydanticTypeParser(),
                StandardTypeParser(),
                DefaultTypeParser(),
            ]

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


class ParameterFormatter:
    def format_parameter_list(self, parameters: list[ParameterInfo]) -> str:
        if not parameters:
            return ''

        formatted_params = []
        for param in parameters:
            formatted_param = self._format_single_parameter(param)
            formatted_params.append(formatted_param)

        return '\n'.join(formatted_params)

    def _format_single_parameter(self, param: ParameterInfo, indent: str = '  ') -> str:
        # base parameter line
        param_line = f'{indent}- {param.name}: {param.type_hint}'

        # add description if available
        if param.description:
            param_line += f' - {param.description}'

        # add required/optional status
        if param.required:
            param_line += ' (required)'
        else:
            default_text = f', default: {param.default}' if param.default is not None else ''
            param_line += f' (optional{default_text})'

        # add constraints
        constraints_text = self._format_constraints(param.constraints)
        if constraints_text:
            param_line += f' {constraints_text}'

        lines = [param_line]

        # add nested fields if available
        if param.nested_fields:
            for nested_param in param.nested_fields:
                nested_line = self._format_single_parameter(nested_param, indent + '    ')
                lines.append(nested_line)

        return '\n'.join(lines)

    def _format_constraints(self, constraints: dict[str, typing.Any]) -> str:
        if not constraints:
            return ''

        formatters = {
            'min_length': lambda v: f'min length: {v}',
            'max_length': lambda v: f'max length: {v}',
            'pattern': lambda v: f'pattern: {v}',
            'examples': self._format_examples,
            'min_value': lambda v: f'min: {v}',
            'max_value': lambda v: f'max: {v}',
        }

        parts = []
        for key, value in constraints.items():
            if key in formatters:
                parts.append(formatters[key](value))

        return f'[{", ".join(parts)}]' if parts else ''

    def _format_examples(self, examples: list) -> str:
        quoted_examples = [f"'{ex}'" for ex in examples]
        return f'examples: {", ".join(quoted_examples)}'
