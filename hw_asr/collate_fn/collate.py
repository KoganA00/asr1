import logging
from typing import List

import torch

logger = logging.getLogger(__name__)


def collate_fn(dataset_items: List[dict]):
    """
    Collate and pad fields in dataset items
    """

    result_batch = {"spectrogram": [],
                    "text_encoded": [],
                    "text_encoded_length": torch.zeros(len(dataset_items)),
                    "text": []}

    # TODO: your code here
    max_len_spec = 0
    max_len_te_en = 0

    specs_list = []
    text_encoded_list = []
    for data in dataset_items:
        max_len_te_en = max(max_len_te_en, data["text_encoded"].shape[-1])
        max_len_spec = max(max_len_spec, data["spectrogram"].shape[-1])

    if len(dataset_items) > 0:
        new_spec_size = torch.tensor(dataset_items[0]["spectrogram"].shape).tolist()

        new_te_en_size = torch.tensor(dataset_items[0]["text_encoded"].shape).tolist()

        result_batch['spectrogram'] = torch.zeros(
            [len(dataset_items)] +
            new_spec_size[1:-1] +
            [max_len_spec])
        result_batch['text_encoded'] = torch.zeros(
            [len(dataset_items)] +
            new_te_en_size[1:-1] +
            [max_len_te_en])

    for ii, data in enumerate(dataset_items):
        new_spec = torch.zeros(new_spec_size, dtype=float)

        result_batch['spectrogram'][ii, :, :data["spectrogram"].shape[-1]] = data["spectrogram"][0]


        result_batch['text_encoded'][ii, :data['text_encoded'].shape[-1]] = data['text_encoded'][0]

        result_batch["text_encoded_length"][ii] = len(data["text_encoded"])
        result_batch["text"].append(data["text"])

    return result_batch
    raise NotImplementedError
