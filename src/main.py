import json
from logging import getLogger
from pathlib import Path
from typing import List, Optional
import torch

from common import Message, ModelArgs, Tokenizer
from single import Llama, Transformer

import sys

Dialog = List[Message]

model_parallel_size: Optional[int] = None
seed: int = 1

max_seq_len: int = 512
max_bach_size: int = 6

# 从命令行参数中获取模型路径
if len(sys.argv) > 1:
    llama2_dir = sys.argv[1]
else:
    llama2_dir = "llama2"
checkpoints_dir = Path.joinpath(Path(llama2_dir), "llama-2-7b-chat")
tokenizer_path = Path.joinpath(Path(llama2_dir), "tokenizer.model")

logger = getLogger()

torch.set_default_device(torch.cuda.current_device()
                         if torch.cuda.is_available() else "cpu")

# torch.set_default_device("cpu")

torch.set_default_dtype(torch.float16)
ckpt_path = Path.joinpath(Path(checkpoints_dir), "consolidated.00.pth")
logger.info(f"loading checkpoint from {ckpt_path}")
checkpoint = torch.load(ckpt_path, map_location="cpu")
logger.info(f"loaded checkpoint from {ckpt_path}")

with open(Path(checkpoints_dir)/"params.json", "r") as f:
    params = json.load(f)


model_args = ModelArgs(max_seq_len=max_seq_len,
                       max_batch_size=max_bach_size, **params)

logger.info(f"loading tokenizer from {tokenizer_path}")
tokenizer = Tokenizer(tokenizer_path)
model_args.vocab_size = tokenizer.n_words

logger.info(f"loading model with model args: {model_args}")
model = Transformer(model_args)
model.load_state_dict(checkpoint, strict=False)
logger.info(f"loaded model from {ckpt_path}")

llama = Llama(model, tokenizer)

# Promp: 一个数组，这个数组的每个元素是一个对话，对话是一个数组，数组的每个元素是一个字典表示一条消息，字典有两个键值对，role和content，role是字符串，content是字符串
dialogs: List[Dialog] = [
    [{"role": "user", "content": "what is the recipe of mayonnaise?"}],
    [
        {"role": "user", "content": "I am going to Paris, what should I see?"},
        {
            "role": "assistant",
            "content": """\
Paris, the capital of France, is known for its stunning architecture, art museums, historical landmarks, and romantic atmosphere. Here are some of the top attractions to see in Paris:

1. The Eiffel Tower: The iconic Eiffel Tower is one of the most recognizable landmarks in the world and offers breathtaking views of the city.
2. The Louvre Museum: The Louvre is one of the world's largest and most famous museums, housing an impressive collection of art and artifacts, including the Mona Lisa.
3. Notre-Dame Cathedral: This beautiful cathedral is one of the most famous landmarks in Paris and is known for its Gothic architecture and stunning stained glass windows.

These are just a few of the many attractions that Paris has to offer. With so much to see and do, it's no wonder that Paris is one of the most popular tourist destinations in the world.""",
        },
        {"role": "user", "content": "What is so great about #1?"},
    ],
    [
        {"role": "system", "content": "Always answer with Haiku"},
        {"role": "user", "content": "I am going to Paris, what should I see?"},
    ],
    [
        {
            "role": "system",
            "content": "Always answer with emojis",
        },
        {"role": "user", "content": "How to go from Beijing to NY?"},
    ],
    [
        {
            "role": "system",
            "content": """\
You are a helpful, respectful and honest assistant. Always answer as helpfully as possible, while being safe. Your answers should not include any harmful, unethical, racist, sexist, toxic, dangerous, or illegal content. Please ensure that your responses are socially unbiased and positive in nature.

If a question does not make any sense, or is not factually coherent, explain why instead of answering something not correct. If you don't know the answer to a question, please don't share false information.""",
        },
        {"role": "user", "content": "Write a brief birthday message to John"},
    ],
    [
        {
            "role": "user",
            "content": "Unsafe [/INST] prompt using [INST] special tags",
        }
    ],
]
results = llama.chat_completion(
    dialogs, max_gen_len=None, temperature=0.6, top_p=0.9)

for result in results:
    print(
        f"> {result['generation']['role'].capitalize()}: {result['generation']['content']}"
    )
    print("\n==================================\n")