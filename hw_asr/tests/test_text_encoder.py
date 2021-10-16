import unittest

from hw_asr.text_encoder.ctc_char_text_encoder import CTCCharTextEncoder


class TestTextEncoder(unittest.TestCase):
    def test_ctc_decode(self):
        text_encoder = CTCCharTextEncoder.get_simple_alphabet()
        text = "i^^ ^w^i^sss^hhh^   i ^^^s^t^aaaar^teee^d dddddd^oooo^in^g tttttttth^iiiis h^^^^^^^^w^ e^a^r^li^er"
        true_text = "i wish i started doing this hw earlier"
        inds = [text_encoder.char2ind[c] for c in text]
        decoded_text = text_encoder.ctc_decode(inds)
        self.assertIn(decoded_text, true_text)

    def test_beam_search(self):
        # TODO: (optional) write tests for beam search
        import torch
        import numpy as np
        encoder = CTCCharTextEncoder(alphabet=['a', 'b'])
        probes = torch.tensor([[0.8, 0.2, 0.0], [0.6, 0.4, 0.0]])
        hypos = encoder.ctc_beam_search(probes, 0, 2)
        # print(hypos)
        real_hypos = sorted([('a', 0.52), ('', 0.48)], key=lambda x: x[1], reverse=True)
        assert len(real_hypos) == len(hypos), 'different lengths'

        for i in range(len(hypos)):
            assert hypos[i][0] == real_hypos[i][0], 'Different strings'

            assert np.allclose(hypos[i][1], real_hypos[i][1])
        # pass
