import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence


class CharRNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(CharRNN, self).__init__()

        self.rnn = nn.RNN(input_size, hidden_size)
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, padded_texts, lengths):
        packed = pack_padded_sequence(padded_texts, lengths.cpu(), enforce_sorted=False)
        _, hidden = self.rnn(packed)
        output = self.h2o(hidden[-1])  # Use final hidden state
        return self.softmax(output)


class CharLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size):
        super(CharLSTM, self).__init__()
        print(
            f"input_size {input_size}, hidden_size {hidden_size}, num_layers {num_layers}, output_size {output_size}"
        )

        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers
        )
        self.h2o = nn.Linear(hidden_size, output_size)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, padded_texts, lengths):
        packed = pack_padded_sequence(padded_texts, lengths.cpu(), enforce_sorted=False)
        _, (hidden, _) = self.lstm(packed)
        output = self.h2o(hidden[-1])  # Use final hidden state
        return self.softmax(output)
