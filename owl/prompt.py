import abc
import pathlib
import string
import textwrap
import tomllib
import typing

import owl.tool as tool


class CustomTemplate(string.Template):
    delimiter = '<$>'

    def get_placeholders(self) -> set[str]:
        return set(self.get_identifiers())


class BasePromptBuilder(abc.ABC):
    def __init__(self, template_str: str = '', template_class: type[string.Template] = CustomTemplate):
        self.template_str = template_str
        self.template_class = template_class
        self._context = {}

    @abc.abstractmethod
    def render(self, **kwargs) -> str:
        pass

    @abc.abstractmethod
    def get_required_arguments(self) -> set[str]:
        pass

    def with_context(self, **kwargs):
        self._context.update(kwargs)
        return self

    def template(self, template_str: str):
        self.template_str = template_str
        return self


class PromptBuilder(BasePromptBuilder):
    def __init__(self, template_str: str = '', template_class: type[string.Template] = CustomTemplate):
        super().__init__(template_str, template_class)

    def render(self, **kwargs) -> str:
        context = {**self._context, **kwargs}
        template = self.template_class(textwrap.dedent(self.template_str).strip())
        return template.substitute(**context)

    def get_required_arguments(self) -> set[str]:
        template = self.template_class(textwrap.dedent(self.template_str).strip())
        return set(template.get_identifiers())

    def with_prompt(self, category: str, name: str, **kwargs):
        template = get_prompt(category, name)
        self.template_str = template
        return self.with_context(**kwargs)

    def with_tools(self, registry: tool.ToolRegistry):
        tools = registry.list_tools()
        tools_description = tool.format_tools_for_prompt(tools)

        tool_instructions = self.with_prompt('system', 'tool_instructions', tools_description=tools_description).render()

        return self.with_context(tool_instructions=tool_instructions, additional_instructions='')


def load_prompts(prompts_file: str | pathlib.Path = None) -> dict[str, typing.Any]:
    if prompts_file is None:
        prompts_file = pathlib.Path(__file__).parent.parent / 'prompts' / 'prompts.toml'

    with open(prompts_file, 'rb') as f:
        return tomllib.load(f)


def get_prompt(category: str, name: str, prompts_file: str | pathlib.Path = None) -> str:
    prompts = load_prompts(prompts_file)

    if category not in prompts:
        raise ValueError(f"Category '{category}' not found")

    if name not in prompts[category]:
        raise ValueError(f"Prompt '{name}' not found in category '{category}'")

    return prompts[category][name]['template']
