import rust_tokenizer;
import tiktoken;
import pickle;
from functools import lru_cache;
from my_nanochat.my_common import get_base_dir
import os
import copy

# copied from https://github.com/karpathy/nanochat/blob/master/nanochat/tokenizer.py
SPLIT_PATTERN = r"""'(?i:[sdmt]|ll|ve|re)|[^\r\n\p{L}\p{N}]?+\p{L}+|\p{N}{1,2}| ?[^\s\p{L}\p{N}]++[\r\n]*|\s*[\r\n]|\s+(?!\S)|\s+"""

SPECIAL_TOKENS = [
    "<|bos|>",
    "<|user_start|>", # user messages
    "<|user_end|>",
    "<|assistant_start|>", # assistant messages
    "<|assistant_end|>",
    "<|python_start|>", # assistant invokes python REPL tool
    "<|python_end|>",
    "<|output_start|>", # python REPL outputs back to assistant
    "<|output_end|>",
]

class MyTokenizer:

    def __init__(self, enc: tiktoken.Encoding):
        self.enc = enc

    @classmethod
    def train_from_iterator(cls, text_iterator, vocab_size):
        tokenizer = rust_tokenizer.Tokenizer()
        vocab_size_no_special = vocab_size - len(SPECIAL_TOKENS) # see what I learned in challenge 23
        tokenizer.train_from_iterator(text_iterator, vocab_size_no_special, buffer_size=8192, pattern=SPLIT_PATTERN);
        mergeable_ranks = tokenizer.get_mergeable_ranks()
        token_offset = len(mergeable_ranks)
        special_tokens = {name: token_offset + i for i, name in enumerate(SPECIAL_TOKENS)}
        return cls(
            enc = tiktoken.Encoding(
                name="my-encoding",
                pat_str=SPLIT_PATTERN,
                mergeable_ranks=dict(mergeable_ranks),
                special_tokens=special_tokens
            )
        )

    def encode(self, text, prepend=None, num_threads=8):

        if prepend is not None:
            assert(isinstance(prepend, int)) # for now at least, can enhance later to accept string or int

        if isinstance(text, str):
            ids = self.enc.encode_ordinary(text)
            if prepend is not None:
                ids.insert(0, prepend)
            return ids
        elif isinstance(text, list):
            batch = self.enc.encode_ordinary_batch(text, num_threads=num_threads)
            if prepend is not None:
                for ids in batch:
                    ids.insert(0, prepend)
            return batch
        else:
            raise ValueError(f"invalid inpuyt type: {type(text)}")

    def decode(self, tokens):
        return self.enc.decode(tokens);

    def save(self, filename):
        with open(filename, 'wb') as f:
            pickle.dump(self.enc, f)
        print(f"Saved tokenizer to {filename}")

    @classmethod
    def load_from_file(cls, filename):
        with open(filename, 'rb') as f:
            enc = pickle.load(f)
        return cls(enc)

    @lru_cache(maxsize=32) # will take his word that encode_single_token is "slow"
    def encode_special(self, text):
        return self.enc.encode_single_token(text)
    
    def get_bos_token_id(self):
        return self.encode_special('<|bos|>') # TODO he decided it was worth it to hold onto it, maybe change to that?

    def get_vocab_size(self):
        return self.enc.n_vocab

    def get_special_tokens(self):
        return self.enc.special_tokens_set

    def render_conversation(self, conversation, max_tokens=2048):
        ids, mask = [], []
        def add_tokens(token_ids, mask_val):
            if isinstance(token_ids, int):
                token_ids = [token_ids]
            ids.extend(token_ids)
            mask.extend([mask_val] * len(token_ids))

        if conversation['messages'][0]['role'] == 'system':
            conversation = copy.deepcopy(conversation)
            messages = conversation['messages']
            assert messages[1]['role'] == 'user'
            messages[1]['content'] = messages[0]['content'] + "\n\n" + messages[1]['content']
            messages = messages[1:]
        else:
            messages = conversation['messages']
        assert len(messages) >= 1

        bos = self.get_bos_token_id()
        user_start, user_end = self.encode_special("<|user_start|>"), self.encode_special("<|user_end|>")
        assistant_start, assistant_end = self.encode_special("<|assistant_start|>"), self.encode_special("<|assistant_end|>")
        python_start, python_end = self.encode_special("<|python_start|>"), self.encode_special("<|python_end|>")
        output_start, output_end = self.encode_special("<|output_start|>"), self.encode_special("<|output_end|>")

        add_tokens(bos, 0)
        for i, message in enumerate(messages):
            
            must_be_from = 'user' if i % 2 == 0 else 'assistant'
            assert message['role'] == must_be_from

            content = message['content']

            if message['role'] == 'user':
                assert isinstance(content, str)
                value_ids = self.encode(content)
                add_tokens(user_start, 0)
                add_tokens(value_ids, 0)
                add_tokens(user_end, 0)
            elif message['role'] == 'assistant':
                add_tokens(assistant_start, 0)
                if isinstance(content, str):
                    value_ids = self.encode(content)
                    add_tokens(value_ids, 1)
                elif isinstance(content, list):
                    for part in content:
                        value_ids = self.encode(part['text'])
                        if part['type'] == 'text':
                            add_tokens(value_ids, 1)
                        elif part['type'] == 'python':
                            add_tokens(python_start, 1)
                            add_tokens(value_ids, 1)
                            add_tokens(python_end, 1)
                        elif part['type'] == 'python_output':
                            add_tokens(output_start, 0)
                            add_tokens(value_ids, 0)
                            add_tokens(output_end, 0)
                        else:
                            assert False
                else:
                    assert False
                add_tokens(assistant_end, 1)
            else:
                assert False

        ids = ids[:max_tokens]
        mask = mask[:max_tokens]
        return ids, mask

    def render_for_completion(self, conversation):
        conversation = copy.deepcopy(conversation)
        messages = conversation['messages']
        assert messages[-1]['role'] == 'assistant'
        messages.pop()

        ids, _ = self.render_conversation(conversation)

        assistant_start = self.encode_special('<|assistant_start|>')
        ids.append(assistant_start)
        return ids

def get_tokenizer():
    return MyTokenizer.load_from_file(os.path.join(get_base_dir(), 'my-tokenizer.pkl'))

def get_token_bytes(device="cpu"):
    import torch
    token_bytes_path = os.path.join(get_base_dir(), "token_bytes.pt")
    assert os.path.exists(token_bytes_path), f"File {token_bytes_path} not found, create it from challenge-18-add-evaluate-bpb/add-evaluate-bpb.ipynb for now" # TODO
    token_bytes = torch.load(token_bytes_path, map_location=device)
    return token_bytes
