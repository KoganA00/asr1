import unittest

from hw_asr.collate_fn.collate import collate_fn
from hw_asr.datasets import LibrispeechDataset
from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder
from hw_asr.utils.parse_config import ConfigParser


import ssl

try:
    _create_unverified_https_context = ssl._create_unverified_context
except AttributeError:
    # Legacy Python that doesn't verify HTTPS certificates by default
    pass
else:
    # Handle target environment that doesn't support HTTPS verification
    ssl._create_default_https_context = _create_unverified_https_context


class TestDataloader(unittest.TestCase):
    def test_collate_fn(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        config_parser = ConfigParser.get_default_configs()

        ds = LibrispeechDataset(
            "dev-clean", text_encoder=text_encoder, config_parser=config_parser
        )

        BS = 3
        print('----')
        thing = [ds[i] for i in range(BS)]
        #print(len(thing))
        #print(thing[0].keys())
        #print('----')
        #print(thing[0]['text'])
        #print(thing[0]['text_encoded'])
        #print(thing[0]['text_encoded'].shape)
        #print(thing[0]['text_encoded'].type())
        #print(thing[0]['spectrogram'].shape)
        #print(thing[0]['spectrogram'].type())

        batch = collate_fn([ds[i] for i in range(BS)])
        #print(batch)

        #print(batch['spectrogram'][0, :, :thing[0]['spectrogram'].shape[-1]] == thing[0]['spectrogram'])

        self.assertIn("spectrogram", batch)  # torch.tensor
        bs, audio_time_length, feature_length = batch["spectrogram"].shape
        self.assertEqual(bs, BS)

        self.assertIn("text_encoded", batch)  # [int] torch.tensor
        # joined and padded indexes representation of transcriptions
        bs, text_time_length = batch["text_encoded"].shape
        self.assertEqual(bs, BS)

        self.assertIn("text_encoded_length", batch)  # [int] torch.tensor
        # contains lengths of each text entry
        self.assertEqual(len(batch["text_encoded_length"].shape), 1)
        bs = batch["text_encoded_length"].shape[0]
        self.assertEqual(bs, BS)

        self.assertIn("text", batch)  # List[str]
        # simple list of initial normalized texts
        bs = len(batch["text"])
        self.assertEqual(bs, BS)

        return batch
