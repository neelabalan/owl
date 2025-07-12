import os
import random
from pathlib import Path

import requests

from owl import agent
from owl import tool
from owl.prompt import PromptBuilder
from owl.prompt import get_prompt


class GPTAgent(agent.Agent):
    def __init__(self, model: str = 'gpt-4o-mini', **kwargs):
        super().__init__(model=model, **kwargs)
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError('OPENAI_API_KEY environment variable is required')

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


def get_weather(location: str) -> str:
    weather_conditions = ['sunny', 'cloudy', 'rainy', 'snowy', 'partly cloudy']
    temperature = random.randint(-10, 35)
    condition = random.choice(weather_conditions)
    return f'Weather in {location}: {condition}, {temperature}°C'


def calculate_math(expression: str) -> str:
    try:
        allowed_chars = set('0123456789+-*/.() ')
        if not all(c in allowed_chars for c in expression):
            return 'Error: Invalid characters in expression'

        result = eval(expression)
        return f'Result: {expression} = {result}'
    except Exception as e:
        return f'Error calculating {expression}: {e}'


def search_files(directory: str, pattern: str) -> str:
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


class ResearchAssistant:
    def __init__(self):
        self.tool_registry = tool.ToolRegistry()
        self._setup_tools()

        system_template = get_prompt('system', 'tool_enabled')
        base_instructions = """You are a helpful research assistant with access to weather, calculation, and file search tools.
You can help with:
- Getting weather information for research planning
- Performing mathematical calculations for data analysis
- Searching for files in directories

Use these tools when they would be helpful for the user's request."""

        prompt_builder = PromptBuilder(system_template)
        complete_instruction = prompt_builder.with_tools(self.tool_registry).render(
            base_instructions=base_instructions, additional_instructions=''
        )
        print(complete_instruction)

        self.agent = GPTAgent(model='gpt-4.1', instruction=complete_instruction, name='Assistant', role=agent.Role.agent)

        self.tool_caller = tool.UniversalToolCaller(self.tool_registry)

    def _setup_tools(self):
        self.tool_registry.register_function(
            get_weather, name='get_weather', description='Get current weather information for any location'
        )
        self.tool_registry.register_function(
            calculate_math, name='calculate_math', description='Perform mathematical calculations safely'
        )
        self.tool_registry.register_function(
            search_files, name='search_files', description='Search for files in a directory using glob patterns'
        )

    async def process_query(self, user_query: str) -> str:
        print(f'User: {user_query}')

        response = self.agent.run(user_query)
        print(f'Assistant: {response}\n\n')

        tool_result = await self.tool_caller.execute_tool_call(response)

        if tool_result.success:
            print(f'Tool executed: {tool_result.result}')

            follow_up_prompt = PromptBuilder().with_prompt('user', 'rag_query',
                query=user_query, doc_count='1', documents=str(tool_result.result)
            ).render()

            final_response = self.agent.run(follow_up_prompt)
            print(f'Final response: {final_response}')
            return final_response
        elif tool_result.error:
            print(f'Tool error: {tool_result.error}')
            return response
        else:
            return response


def run_demo():
    test_queries = [
        "What's the weather like in Tokyo?",
        'Calculate 15% of 250',
        'Find all Python files in the current directory',
        "What's the weather in London and calculate the tip for a 45 dollar meal at 18%?",
    ]

    assistant = ResearchAssistant()

    print('Research Assistant Demo')
    print('Testing tool-enabled assistant with weather, calculator, and file search\n')

    import asyncio

    async def run_tests():
        for i, query in enumerate(test_queries, 1):
            print(f'Test {i}: {query}')
            await assistant.process_query(query)
            print('-' * 50)

    asyncio.run(run_tests())


async def interactive_mode():
    assistant = ResearchAssistant()

    print('Interactive Research Assistant')
    print('Available tools: weather, calculator, file search')
    print("Type 'quit' to exit\n")

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


if __name__ == '__main__':
    # import asyncio
    # asyncio.run(interactive_mode())

    run_demo()
