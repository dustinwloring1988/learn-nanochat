import os
import json
from my_tasks.my_common import MyTask

class MyCustomJSON(MyTask):

    def __init__(self, filepath, **kwargs):
        super().__init__(**kwargs)
        self.filepath = filepath
        self.conversations = []

        with open(filepath, 'r', encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                messages = json.loads(line)
                assert isinstance(messages, list)
                assert len(messages) >= 2
                for i, message in enumerate(messages):
                    assert 'role' in message
                    assert 'content' in message
                    expected_role = 'user' if i % 2 == 0 else 'assistant'
                    assert message['role'] == expected_role
                    assert isinstance(message['content'], str)

                self.conversations.append(messages)

        self.length = len(self.conversations)

    def num_examples(self):
        return self.length

    def get_example(self, index):
        messages = self.conversations[index]
        conversation = {
            'messages': messages
        }
        return conversation