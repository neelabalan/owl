import json
import os

import requests

from owl import agent


class GemmaAgent(agent.Agent):
    _url = os.getenv("OLLAMA_URL")

    def run(self, prompt: str):
        try:
            response = requests.post(self._url, headers={'Content-Type': 'application/json'}, data=json.dumps({'model': self.model, 'prompt': prompt, 'stream': False}))
            response.raise_for_status()

            json_response = response.json()
            return json_response['response']

        except requests.exceptions.RequestException as e:
            print(f'Error communicating with Ollama: {e}')
            return None
        except (KeyError, TypeError) as e:
            print(f'Error parsing Ollama response: {e}')
            return None


class HumanInTheLoopConversation(agent.Conversation):
    # better to be an even number?
    _max_history: int = 10

    def construct_message_history(self) -> str:
        if not self.thread.messages:
            return ""
        prompt = '< Conversation history >\n'
        for message in self.thread.messages[-self._max_history :]:
            prompt += f'{message.author}: {message.content}\n'
        # prompt += "Your turn: "
        return prompt

    def engage(self) -> agent.Message:
        response = ''
        for participant in self.participants:
            if isinstance(participant, agent.Agent):
                response = agent.Message(author=participant.name, content=participant.run(f"System instruction: {participant.instruction}\n" + self.construct_message_history()))
                print(f"{participant.name}: {response.content}")
            elif isinstance(participant, agent.Human):
                response = agent.Message(author=participant.name, content=input('User: '))
            else:
                pass
            self.thread.add_message(response)
        return response


def run():
    participants = [agent.Human(name='george', role=agent.Role.user), GemmaAgent(model="gemma3:27b", instruction='You are a poet', name='Frost', role=agent.Role.agent)]
    conversation = HumanInTheLoopConversation(participants)
    while True:
        try:
            response = conversation.engage()
        except KeyboardInterrupt as _:
            print('keyboard interrupt...')
            break


if __name__ == '__main__':
    run()
