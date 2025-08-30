import json
import os

import httpx

from owl import agent


class OllamaAgent(agent.Agent):
    _url = os.getenv('OLLAMA_URL')

    def run(self, prompt: str):
        try:
            # Use the correct Ollama chat API format
            payload = {'model': self.model, 'messages': [{'role': 'user', 'content': prompt}], 'stream': False}

            with httpx.Client() as client:
                response = client.post(
                    self._url,
                    headers={'Content-Type': 'application/json'},
                    content=json.dumps(payload),
                )
                response.raise_for_status()

                json_response = response.json()
                return json_response['message']['content']

        except httpx.RequestError as e:
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
            return ''
        prompt = '< Conversation history >\n'
        for message in self.thread.messages[-self._max_history :]:
            prompt += f'{message.author}: {message.content}\n'
        # prompt += "Your turn: "
        return prompt

    def engage(self) -> agent.Message:
        response = ''
        for participant in self.participants:
            if isinstance(participant, agent.Agent):
                response = agent.Message(
                    author=participant.name,
                    content=participant.run(
                        f'System instruction: {participant.instruction}\n' + self.construct_message_history()
                    ),
                )
                print(f'{participant.name}: {response.content}')
            elif isinstance(participant, agent.Human):
                response = agent.Message(author=participant.name, content=input('User: '))
            else:
                pass
            self.thread.add_message(response)
        return response


class MasterAgentConversation(agent.Conversation):
    def __init__(self, participants: list[agent.Entity], master_agent: agent.Agent, max_history: int = 10):
        super().__init__(participants)
        self.master_agent = master_agent
        self._max_history = max_history
        self._turn_count = 0

    def construct_message_history(self) -> str:
        if not self.thread.messages:
            return ''
        prompt = '< Conversation history >\n'
        for message in self.thread.messages[-self._max_history :]:
            prompt += f'{message.author}: {message.content}\n'
        return prompt

    def construct_participant_list(self) -> str:
        participant_info = []
        for i, participant in enumerate(self.participants):
            if isinstance(participant, agent.Agent):
                participant_info.append(f'{i + 1}. {participant.name} (Agent) - {participant.instruction}')
            elif isinstance(participant, agent.Human):
                participant_info.append(f'{i + 1}. {participant.name} (Human)')
        return '\n'.join(participant_info)

    def decide_next_speaker(self) -> agent.Entity:
        """Use the master agent to decide who should speak next"""
        participants_list = self.construct_participant_list()
        conversation_history = self.construct_message_history()

        master_prompt = f"""You are an expert medical research coordinator moderating a problem-solving session. Based on the conversation history and participant expertise, decide who should contribute next.

TEAM MEMBERS:
{participants_list}

{conversation_history}

Moderation Guidelines:
1. Consider each specialist's expertise and when their input would be most valuable
2. Ensure molecular, clinical, and technical perspectives are all represented
3. When one discipline's insights could inform another, bring in that expert
4. If the discussion needs practical implementation details, involve the medical technician
5. If clinical implications need consideration, involve the medical doctor
6. For biological mechanisms and research approaches, involve the molecular biologist

CRITICAL: You must respond with ONLY a single number (1, 2, or 3) and absolutely nothing else. No explanation, no thinking, no additional text - just the number.

Who should contribute next?"""

        try:
            response = self.master_agent.run(master_prompt)
            if response:
                # Try to extract number from response, handling various formats
                import re

                numbers = re.findall(r'\b([123])\b', response)
                if numbers:
                    participant_number = int(numbers[0])
                    if 1 <= participant_number <= len(self.participants):
                        print(
                            f'[Coordinator] Selected participant {participant_number}: {self.participants[participant_number - 1].name}'
                        )
                        return self.participants[participant_number - 1]
        except (ValueError, IndexError):
            pass

        print(f'Master agent gave invalid response: {response}. Falling back to round-robin.')

        # Fallback to round-robin if master agent fails
        return self.participants[self._turn_count % len(self.participants)]

    def engage(self) -> agent.Message:
        # Decide who should speak next
        next_speaker = self.decide_next_speaker()
        self._turn_count += 1

        response = None
        if isinstance(next_speaker, agent.Agent):
            content = next_speaker.run(
                f'System instruction: {next_speaker.instruction}\n' + self.construct_message_history()
            )
            response = agent.Message(author=next_speaker.name, content=content)
            print(f'{next_speaker.name}: {response.content}')
        elif isinstance(next_speaker, agent.Human):
            content = input(f'{next_speaker.name}: ')
            response = agent.Message(author=next_speaker.name, content=content)

        if response:
            self.thread.add_message(response)

        return response


def run_with_master_agent():
    # Create medical team participants
    participants = [
        OllamaAgent(
            model='gemma3:27b',
            instruction='You are a molecular biologist specializing in cellular mechanisms, protein interactions, and genetic research. You approach problems from a fundamental biological perspective.',
            name='Dr. Chen',
            role=agent.Role.agent,
        ),
        OllamaAgent(
            model='phi4:latest',
            instruction='You are an experienced medical doctor with expertise in diagnostics, treatment protocols, and patient care. You focus on clinical applications and patient outcomes.',
            name='Dr. Rodriguez',
            role=agent.Role.agent,
        ),
        OllamaAgent(
            model='mistral-small3.2:latest',
            instruction='You are a medical technician with deep knowledge of laboratory procedures, diagnostic equipment, and practical implementation of medical protocols.',
            name='Tech Williams',
            role=agent.Role.agent,
        ),
    ]

    # Create an expert master agent/moderator who knows the medical domain
    master_agent = OllamaAgent(
        model='deepseek-r1:8b',
        instruction='You are an expert medical research coordinator who understands molecular biology, clinical medicine, and laboratory procedures. You moderate discussions to ensure all perspectives are heard and guide the team toward solving complex medical problems effectively.',
        name='Coordinator',
        role=agent.Role.agent,
    )

    conversation = MasterAgentConversation(participants, master_agent)
    print('Medical Research Team Collaboration')
    print('Team Members:')
    print('Dr. Chen - Molecular Biologist')
    print('Dr. Rodriguez - Medical Doctor')
    print('Tech Williams - Medical Technician')
    print('Coordinator - Research Coordinator (Moderator)')
    print('\nThe team will collaborate to solve medical/biological problems.')
    print('The coordinator will intelligently decide who contributes next.')
    print('Press Ctrl+C to exit.\n')

    # Start with an initial problem for the team to solve
    initial_problem = 'A new viral infection is causing unusual symptoms: patients develop severe fatigue, cognitive impairment, and abnormal protein aggregation in neural tissue. The virus seems to affect cellular protein folding mechanisms. How should we approach understanding and treating this condition?'

    initial_message = agent.Message(author='Research Prompt', content=initial_problem)
    conversation.thread.add_message(initial_message)
    print(f'🔬 Initial Problem: {initial_problem}\n')

    while True:
        try:
            conversation.engage()
        except KeyboardInterrupt:
            print('\n\n🔬 Research session ended.')
            break


def run():
    # participants = [agent.Human(name='george', role=agent.Role.user), OllamaAgent(model="gemma3:27b", instruction='You are a philosopher', name='Frost', role=agent.Role.agent)]
    participants = [
        agent.Human(name='george', role=agent.Role.user),
        OllamaAgent(model='gemma3:27b', instruction='You are a philosopher', name='Speaker', role=agent.Role.agent),
        OllamaAgent(model='phi4:latest', instruction='You are a scientist', name='Brainy', role=agent.Role.agent),
    ]
    conversation = HumanInTheLoopConversation(participants)
    while True:
        try:
            conversation.engage()
        except KeyboardInterrupt as _:
            print('keyboard interrupt...')
            break


if __name__ == '__main__':
    run_with_master_agent()
    # run()
