"""
Tokenization transforms for VLM data processing.

Shared by the Zephon and Grain preprocessing pipelines.
"""

from typing import TYPE_CHECKING, List

if TYPE_CHECKING:
    from transformers import PreTrainedTokenizer

    from .models import Conversation


def tokenize_conversations(
    conversations: List["Conversation"],
    tokenizer: "PreTrainedTokenizer",
) -> None:
    """Batch tokenize all messages in conversations for efficiency.


    Mutates conversations in place, setting token_ids on each message.

    Args:
        conversations: List of Conversation objects to tokenize
        tokenizer: HuggingFace tokenizer
    """
    # Collect texts and track assignment locations
    texts_to_tokenize = []
    assignment_map = []  # (conv_idx, msg_idx) tuples

    for conv_idx, conversation in enumerate(conversations):
        for msg_idx, message in enumerate(conversation.messages):
            texts_to_tokenize.append(message.content)
            assignment_map.append((conv_idx, msg_idx))

    if texts_to_tokenize:
        # Batch tokenization is much faster than individual calls
        batch_token_ids = tokenizer(
            texts_to_tokenize,
            add_special_tokens=False,  # Special tokens handled by model
            return_attention_mask=False,
            return_token_type_ids=False,
            verbose=False,
        )["input_ids"]

        # Assign results back to messages
        for token_ids, (conv_idx, msg_idx) in zip(batch_token_ids, assignment_map):
            conversations[conv_idx].messages[msg_idx].token_ids = token_ids
