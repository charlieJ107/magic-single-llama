from typing import List, Optional, Tuple
import torch

from single import Transformer
from common import ChatPrediction, CompletionPrediction, Message,  Tokenizer, sample_top_p

Dialog = List[Message]

B_INST, E_INST = "[INST]", "[/INST]"
B_SYS, E_SYS = "<<SYS>>\n", "\n<</SYS>>\n\n"

SPECIAL_TAGS = [B_INST, E_INST, "<<SYS>>", "<</SYS>>"]
UNSAFE_ERROR = "Error: special tags are not allowed as part of the prompt."


class Llama:
    def __init__(self, model: Transformer, tokenizer: Tokenizer):
        self.model = model
        self.tokenizer = tokenizer

    @torch.inference_mode()
    def generate(
        self,
        prompt_tokens: List[List[int]],
        max_gen_len: int,
        temperature: float = 0.6,
        top_p: float = 0.9,
        echo: bool = False,
    ) -> Tuple[List[List[int]], Optional[List[List[float]]]]:
        """
        Generate text sequences based on provided prompts using the language generation model.

        Args:
            prompt_tokens (List[List[int]]): List of tokenized prompts, where each prompt is represented as a list of integers.
            max_gen_len (int): Maximum length of the generated text sequence.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            Tuple[List[List[int]], Optional[List[List[float]]]]: A tuple containing generated token sequences and, if logprobs is True, corresponding token log probabilities.

        Note:
            This method uses the provided prompts as a basis for generating text. It employs nucleus sampling to produce text with controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        params = self.model.params
        bsz = len(prompt_tokens)  # How many dialogs are there
        assert bsz <= params.max_batch_size, (bsz, params.max_batch_size)

        # lenth of the shortest dialog
        min_prompt_len = min(len(t) for t in prompt_tokens)

        max_prompt_len = max(len(t) for t in prompt_tokens)
        assert max_prompt_len <= params.max_seq_len

        # length of the longest dialog after generation
        total_len = min(params.max_seq_len, max_gen_len + max_prompt_len)

        pad_id = self.tokenizer.pad_id
        # tokens: a container for the entire content, from the prompt to the generated content
        # tokens is a 2d tensor, the first dimension is the dialog index, the second dimension is the token index
        tokens = torch.full((bsz, total_len), pad_id,
                            dtype=torch.long)
        for index, value in enumerate(prompt_tokens):
            # for each dialog, fill the prompt tokens into the tokens tensor
            tokens[index, : len(value)] = torch.tensor(
                value, dtype=torch.long)

        prev_pos = 0
        eos_reached = torch.tensor([False] * bsz)
        input_text_mask = tokens != pad_id
        if min_prompt_len == total_len:  # No space for generation
            logits = self.model.forward(tokens, prev_pos)

        for cur_pos in range(min_prompt_len, total_len):
            # cur_pos: Self regression,
            # we generate the next token beased on token on current position
            # so the current position should range from the minimum length of the prompt, which where the genereated content starts,
            # to the total length of the prompt, which is the maximum length of the prompt

            # tokens[:, prev_pos:cur_pos]:
            # the tokens has multiple dialogs, so the first [:, ...] is the dialog index, the second [..., prev_pos:cur_pos] is the token index
            logits = self.model.forward(tokens[:, prev_pos:cur_pos], prev_pos)
            # logits.shape: (bsz, cur_pos - prev_pos, vocab_size)
            if temperature > 0:
                probs = torch.softmax(logits[:, -1] / temperature, dim=-1)
                # probs.shape: (bsz, vocab_size)

                # sample_top_p: sample from the top_p probability distribution
                # probs: the probability distribution of the last token
                # get the next token
                next_token = sample_top_p(probs, top_p)
            else:
                next_token = torch.argmax(logits[:, -1], dim=-1)
            # next_token.shape: (bsz, 1)
            next_token = next_token.reshape(-1)
            # next_token.shape: (bsz,)
            # only replace token if prompt has already been generated
            next_token = torch.where(
                input_text_mask[:, cur_pos], tokens[:, cur_pos], next_token
            )
            tokens[:, cur_pos] = next_token
            eos_reached |= (~input_text_mask[:, cur_pos]) & (
                next_token == self.tokenizer.eos_id
            )
            prev_pos = cur_pos
            if all(eos_reached):
                break
        out_tokens = []
        for i, toks in enumerate(tokens.tolist()):
            # cut to max gen len
            start = 0 if echo else len(prompt_tokens[i])
            toks = toks[start: len(prompt_tokens[i]) + max_gen_len]
            probs = None
            # cut to eos tok if any
            if self.tokenizer.eos_id in toks:
                eos_idx = toks.index(self.tokenizer.eos_id)
                toks = toks[:eos_idx]
            out_tokens.append(toks)
        return out_tokens

    def text_completion(
        self,
        prompts: List[str],
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None,
        echo: bool = False,
    ) -> List[CompletionPrediction]:
        """
        Perform text completion for a list of prompts using the language generation model.

        Args:
            prompts (List[str]): List of text prompts for completion.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated completion sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.
            echo (bool, optional): Flag indicating whether to include prompt tokens in the generated output. Defaults to False.

        Returns:
            List[CompletionPrediction]: List of completion predictions, each containing the generated text completion.

        Note:
            This method generates text completions for the provided prompts, employing nucleus sampling to introduce controlled randomness.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = [self.tokenizer.encode(
            x, bos=True, eos=False) for x in prompts]
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p,
            echo=echo,
        )

        return [{"generation": self.tokenizer.decode(t)} for t in generation_tokens]

    def chat_completion(
        self,
        dialogs: Dialog,
        temperature: float = 0.6,
        top_p: float = 0.9,
        max_gen_len: Optional[int] = None
    ) -> List[ChatPrediction]:
        """
        Generate assistant responses for a list of conversational dialogs using the language generation model.

        Args:
            dialogs (List[Dialog]): List of conversational dialogs, where each dialog is a list of messages.
            temperature (float, optional): Temperature value for controlling randomness in sampling. Defaults to 0.6.
            top_p (float, optional): Top-p probability threshold for nucleus sampling. Defaults to 0.9.
            max_gen_len (Optional[int], optional): Maximum length of the generated response sequence.
                If not provided, it's set to the model's maximum sequence length minus 1.
            logprobs (bool, optional): Flag indicating whether to compute token log probabilities. Defaults to False.

        Returns:
            List[ChatPrediction]: List of chat predictions, each containing the assistant's generated response.

        Raises:
            AssertionError: If the last message in a dialog is not from the user.
            AssertionError: If the dialog roles are not in the required 'user', 'assistant', and optional 'system' order.

        Note:
            This method generates assistant responses for the provided conversational dialogs.
            It employs nucleus sampling to introduce controlled randomness in text generation.
            If logprobs is True, token log probabilities are computed for each generated token.

        """
        if max_gen_len is None:
            max_gen_len = self.model.params.max_seq_len - 1
        prompt_tokens = []
        unsafe_requests = []
        for dialog in dialogs:
            # 检查对话中是否包含特殊标签-
            unsafe_requests.append(
                any([tag in msg["content"]
                    for tag in SPECIAL_TAGS for msg in dialog])
            )

            # 把对话中的“system”角色的消息拼接到“user”角色的消息前面, 用E_SYS和B_SYS分隔
            # 整个对话变成第一个消息是user的消息， 但是前面加了system的消息，然后是assistant的消息
            if dialog[0]["role"] == "system":
                dialog = [
                    {
                        "role": dialog[1]["role"],
                        "content": B_SYS
                        + dialog[0]["content"]
                        + E_SYS
                        + dialog[1]["content"],
                    }
                ] + dialog[2:]
            # 对话是一个user, assistant, user, assistant, user, assistant...的顺序
            assert all([msg["role"] == "user" for msg in dialog[::2]]) and all(
                [msg["role"] == "assistant" for msg in dialog[1::2]]
            ), (
                "model only supports 'system', 'user' and 'assistant' roles, "
                "starting with 'system', then 'user' and alternating (u/a/u/a/u...)"
            )
            # 对于多轮会话，（从第二个消息开始）将用户的提示词和回答拼接成一个字符串，然后编码
            dialog_tokens: List[int] = sum(
                [
                    self.tokenizer.encode(
                        f"{B_INST} {(prompt['content']).strip()} {E_INST} {(answer['content']).strip()} ",
                        bos=True,
                        eos=True,
                    )
                    for prompt, answer in zip(
                        dialog[::2],
                        dialog[1::2],
                    )
                ],
                [],
            )
            assert (
                dialog[-1]["role"] == "user"
            ), f"Last message must be from user, got {dialog[-1]['role']}"

            # 倒数第一个会话（如果会话中只有用户问题，就仅包含有用户消息），
            # 将用户的提示词拼接成一个字符串，然后编码
            dialog_tokens += self.tokenizer.encode(
                f"{B_INST} {(dialog[-1]['content']).strip()} {E_INST}",
                bos=True,
                eos=False,
            )
            prompt_tokens.append(dialog_tokens)
            # Prompt tokens: 一个二维数组，m个对话，每个对话是一个一维数组，表示对话所有文本tokenzied后得到的tokens
        generation_tokens = self.generate(
            prompt_tokens=prompt_tokens,
            max_gen_len=max_gen_len,
            temperature=temperature,
            top_p=top_p
        )
        return [
            {
                "generation": {
                    "role": "assistant",
                    "content": self.tokenizer.decode(t)
                    # if not unsafe else UNSAFE_ERROR,
                }
            }
            # for t, unsafe in zip(generation_tokens, unsafe_requests)
            for t in generation_tokens
        ]
