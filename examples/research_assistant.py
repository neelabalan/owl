import dataclasses
import os
import random
from pathlib import Path
from typing import Annotated

import pydantic
import requests

from owl import agent
from owl import tool
from owl.prompt import PromptBuilder


class TaskAssignee(pydantic.BaseModel):
    name: str = pydantic.Field(description='Name of the assignee')
    email: str = pydantic.Field(description='Email address', pattern=r'^[^@]+@[^@]+\.[^@]+$')
    role: str = pydantic.Field(description='Role or department', default='developer')


class TaskRequest(pydantic.BaseModel):
    title: str = pydantic.Field(description='Brief title for the task')
    priority: str = pydantic.Field(description='Task priority level', pattern='^(low|medium|high|urgent)$')
    deadline: str = pydantic.Field(description='Deadline in YYYY-MM-DD format', pattern=r'^\d{4}-\d{2}-\d{2}$')
    category: str = pydantic.Field(description='Task category', default='general')
    # this works
    # assignee: TaskAssignee = pydantic.Field(description='Person assigned to this task')
    tags: list[str] = pydantic.Field(description='List of tags for categorization', default=[])


def create_task(task_data: TaskRequest) -> str:
    """Create and schedule a new task with validation"""
    try:
        # Simulate task creation
        task_id = random.randint(1000, 9999)

        # Format tags
        tags_str = ', '.join(task_data.tags) if task_data.tags else 'None'

        response = f"""Task created successfully!
Task ID: {task_id}
Title: {task_data.title}
Priority: {task_data.priority.upper()}
Deadline: {task_data.deadline}
Category: {task_data.category}
Tags: {tags_str}
Status: Scheduled"""

        return response
    except Exception as e:
        return f'Error creating task: {e}'


def get_weather(location: str) -> str:
    weather_conditions = ['sunny', 'cloudy', 'rainy', 'snowy', 'partly cloudy']
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    return f'Weather in {location}: {condition}, {temperature}°C'


def calculate_math(expression: str) -> str:
    """Perform mathematical calculations safely"""
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return 'Error: Invalid characters in expression'

        result = eval(expression)
        return f'Result: {expression} = {result}'
    except Exception as e:
        return f'Error calculating {expression}: {e}'


def search_files(
    directory: Annotated[str, 'Directory path to search in', {'examples': ['.', '/home/user', 'src/']}],
    pattern: Annotated[str, 'Glob pattern for file matching', {'examples': ['*.py', '**/*.txt', 'test_*.py']}],
) -> str:
    """search for files in a directory using glob pataterns"""
    try:
        dir_path = Path(directory)
        if not dir_path.exists():
            return f'Directory {directory} does not exist'

        matching_files = list(dir_path.glob(pattern))
        if not matching_files:
            return f"No files matching '{pattern}' found in {directory}"

        file_list = '\n'.join([str(f.name) for f in matching_files[:10]])
        count = len(matching_files)
        return f"Found {count} files matching '{pattern}' in {directory}:\n{file_list}"
    except Exception as e:
        return f'Error searching files: {e}'


def create_research_tools() -> tool.ToolRegistry:
    registry = tool.ToolRegistry()
    registry.register(calculate_math)
    registry.register(get_weather, name='get_weather', description='Get current weather information for any location')
    registry.register(search_files)
    registry.register(create_task, description='Create and schedule a new task with structured validation')
    return registry


def create_research_instruction(tool_registry: tool.ToolRegistry) -> str:
    from owl.prompt import get_prompt

    base_instructions = """You are a helpful research assistant with access to weather, calculation, file search, and task management tools.
You can help with:
- Getting weather information for research planning
- Performing mathematical calculations for data analysis
- Searching for files in directories
- Creating and scheduling tasks with structured validation

Use these tools when they would be helpful for the user's request."""

    system_template = get_prompt('system', 'tool_enabled')
    return (
        PromptBuilder(system_template)
        .with_tools(tool_registry)
        .render(base_instructions=base_instructions, additional_instructions='')
    )


@dataclasses.dataclass
class ResearchAgent(agent.Agent):
    def __post_init__(self):
        api_key = os.getenv('OPENAI_API_KEY')
        if not api_key:
            raise ValueError('OPENAI_API_KEY environment variable is required')

        self.api_key = api_key
        self.tool_caller = tool.UniversalToolCaller(self.tool_registry)

    def run(self, prompt: str) -> str:
        try:
            messages = []

            if self.instruction:
                messages.append({'role': 'system', 'content': self.instruction})

            messages.append({'role': 'user', 'content': prompt})

            response = requests.post(
                'https://api.openai.com/v1/chat/completions',
                headers={'Authorization': f'Bearer {self.api_key}', 'Content-Type': 'application/json'},
                json={
                    'model': self.model,
                    'messages': messages,
                    'temperature': 0.3,
                    'max_tokens': 1000,
                },
            )
            response.raise_for_status()

            json_response = response.json()
            return json_response['choices'][0]['message']['content']

        except requests.exceptions.RequestException as e:
            print(f'Error communicating with OpenAI: {e}')
            return f'Error: {e}'
        except (KeyError, TypeError) as e:
            print(f'Error parsing OpenAI response: {e}')
            return f'Error: {e}'

    async def process_query(self, user_query: str) -> str:
        print(f'User: {user_query}')

        response = self.run(user_query)
        print(f'Assistant: {response}\n')

        # check if the initial response contains a tool call
        if not self.tool_caller.is_tool_call(response):
            return response

        # keep processing tool calls until no more are found
        tool_results = []
        current_response = response
        max_iterations = 5
        iteration = 0

        while iteration < max_iterations:
            tool_result = self.tool_caller.execute_tool_call(current_response)

            if tool_result.success:
                print(f'Tool executed: {tool_result.result}')
                tool_results.append(tool_result.result)

                # step 1: Ask if assistant needs more tools
                if len(tool_results) == 1:
                    context_info = "You have executed 1 tool so far."
                else:
                    context_info = f"You have executed {len(tool_results)} tools so far."

                decision_prompt = (
                    PromptBuilder()
                    .with_prompt('user', 'tool_followup')
                    .render(
                        user_query=user_query,
                        tool_result=tool_result.result,
                        context=context_info
                    )
                )

                decision_response = self.run(decision_prompt).strip().upper()
                print(f'Assistant decision: {decision_response}')

                if decision_response.lower().startswith('yes'):
                    # step 2: Ask for the specific tool call
                    tool_request_prompt = (
                        PromptBuilder()
                        .with_prompt('user', 'tool_request')
                        .render(user_query=user_query, tool_result=tool_result.result)
                    )

                    current_response = self.run(tool_request_prompt)
                    print(f'Assistant tool request: {current_response}\n')
                    iteration += 1
                elif decision_response.lower().startswith('no'):
                    print('Assistant indicated no more tools needed')
                    break
                else:
                    print(f'Unclear decision response: {decision_response}')
                    break

            elif tool_result.error:
                print(f'Tool error: {tool_result.error}')
                break
            else:
                # no tool call found, this is the final response
                break

        # provide final comprehensive response
        if tool_results:
            tool_results_formatted = '\n'.join(f'- {result}' for result in tool_results)
            final_prompt = (
                PromptBuilder()
                .with_prompt('user', 'final_response')
                .render(user_query=user_query, tool_results=tool_results_formatted)
            )

            final_response = self.run(final_prompt)
            print(f'Final response: {final_response}')
            return final_response
        else:
            return current_response


def run_demo(interactive: bool = False):
    # Setup tools and instructions outside the agent
    research_tools = create_research_tools()
    research_instruction = create_research_instruction(research_tools)

    # see what the instruction looks like
    # print(research_instruction)
    # return

    assistant = ResearchAgent(
        instruction=research_instruction,
        name='ResearchAgent',
        role=agent.Role.agent,
        model='gpt-4.1',
        tool_registry=research_tools,
    )

    if interactive:
        print('Interactive Research Assistant')
        print('Available tools: weather, calculator, file search, task creation')
        print("Type 'quit' to exit\n")

        import asyncio

        async def interactive_loop():
            while True:
                try:
                    user_input = input('\nYour query: ').strip()
                    if user_input.lower() in ['quit', 'exit', 'q']:
                        print('Goodbye!')
                        break

                    if user_input:
                        await assistant.process_query(user_input)
                except KeyboardInterrupt:
                    print('\nGoodbye!')
                    break

        asyncio.run(interactive_loop())
    else:
        test_queries = [
            "Hi How are you?",
            "What do you think about AI? explain in briefly in one sentence.",
            "What's the weather like in Tokyo?",
            'Calculate 15% of 250',
            'Find all Python files in the current directory',
            'Create a todo for me. I need complete my research paper before July 13 2025. It is very important.',
            'Calculate 15% of 250 and 49.13% of 29807',
        ]

        print('Research Assistant Demo')
        print('Testing tool-enabled assistant with weather, calculator, and file search\n')

        import asyncio

        async def run_tests():
            for i, query in enumerate(test_queries, 1):
                print(f'Test {i}: {query}')
                await assistant.process_query(query)
                print('-' * 50)

        asyncio.run(run_tests())


if __name__ == '__main__':
    # Run in demo mode by default
    run_demo(interactive=False)

    # Uncomment the line below to run in interactive mode instead
    # run_demo(interactive=True)
