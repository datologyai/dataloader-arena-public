"""Small local datasets and tokenizer shared by loader and checkpoint tests."""

import io
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))


def preparer():
    import prepare_data

    return prepare_data


@pytest.fixture
def text_data(tmp_path):
    prep = preparer()
    directory = tmp_path / "text"
    directory.mkdir()
    writer = prep.LitWriter(directory, "text", 1024)
    for index in range(16):
        values = np.arange(index * 8, index * 8 + 8, dtype="<u4")
        writer.add(prep.encode_leaves([values.tobytes()]))
    writer.finish()
    return directory


@pytest.fixture
def vlm_data(tmp_path):
    prep = preparer()
    directory = tmp_path / "vlm"
    directory.mkdir()
    source = directory / ".vlm-records"
    source.mkdir()
    image = io.BytesIO()
    Image.new("RGB", (16, 16), "blue").save(image, format="PNG")
    writer = prep.LitWriter(source, "vlm", 8192)
    provenance = []
    for index in range(8):
        row = {
            "_images": [image.getvalue()],
            "_source_id": f"fixture-{index}",
            "conversations": [
                {"from": "human", "value": "<image> Describe this image."},
                {"from": "gpt", "value": "A blue square."},
            ],
        }
        encoded = prep.encode_vlm_row(row)
        assert encoded is not None
        writer.add(encoded)
        provenance.append({"index": index, "source": row["_source_id"]})
    writer.finish()
    (source / "sources.jsonl").write_text(
        "".join(json.dumps(row) + "\n" for row in provenance)
    )
    prep.shuffle_vlm_export(source, directory, 42, 8192)
    shutil.rmtree(source)
    return directory


def local_tokenizer():
    from tokenizers import Tokenizer, models, pre_tokenizers
    from transformers import PreTrainedTokenizerFast

    tokens = [
        "[UNK]",
        "[PAD]",
        "<|vision_start|>",
        "<|vision_end|>",
        "<|image_pad|>",
        "<|im_start|>",
        "<|im_end|>",
    ]
    tokenizer = Tokenizer(
        models.WordLevel(
            {token: index for index, token in enumerate(tokens)}, unk_token="[UNK]"
        )
    )
    tokenizer.pre_tokenizer = pre_tokenizers.Whitespace()
    return PreTrainedTokenizerFast(
        tokenizer_object=tokenizer,
        unk_token="[UNK]",
        pad_token="[PAD]",
        additional_special_tokens=tokens[2:],
    )
