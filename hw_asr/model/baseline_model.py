from torch import nn
from torch.nn import Sequential

from hw_asr.base import BaseModel


class BaselineModel(BaseModel):
    def __init__(self, n_feats, n_class, fc_hidden=512, *args, **kwargs):
        super().__init__(n_feats, n_class, *args, **kwargs)
        #self.net = Sequential(
            # people say it can aproximate any function...
         #   nn.Linear(in_features=n_feats, out_features=fc_hidden),
         #   nn.ReLU(),
         #   nn.Linear(in_features=fc_hidden, out_features=fc_hidden),
          #  nn.ReLU(),
           # nn.Linear(in_features=fc_hidden, out_features=n_class)
        #)
        self.rnn = nn.RNN(input_size = n_feats, hidden_size=128, num_layers=6)
        self.out = nn.Linear(in_features=128, out_features=n_class)



    def forward(self, spectrogram, *args, **kwargs):
        output, _ = self.rnn(spectrogram)
        return {"logits": self.out(output)}

    def transform_input_lengths(self, input_lengths):
        return input_lengths  # we don't reduce time dimension here
