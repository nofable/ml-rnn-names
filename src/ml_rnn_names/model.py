import torch
import torch.nn as nn
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


class GenRNN(nn.Module):
    def __init__(self, n_categories, input_size, hidden_size, output_size):
        super(GenRNN, self).__init__()
        self.hidden_size = hidden_size

        self.i2h = nn.Linear(n_categories + input_size + hidden_size, hidden_size)
        self.i2o = nn.Linear(n_categories + input_size + hidden_size, output_size)
        self.o2o = nn.Linear(hidden_size + output_size, output_size)
        self.dropout = nn.Dropout(0.1)
        self.softmax = nn.LogSoftmax(dim=1)

    def forward(self, category, inpt, hidden):
        input_combined = torch.cat((category, inpt, hidden), 1)
        hidden = self.i2h(input_combined)
        output = self.i2o(input_combined)
        output_combined = torch.cat((hidden, output), 1)
        output = self.o2o(output_combined)
        output = self.dropout(output)
        output = self.softmax(output)
        return output, hidden

    def initHidden(self):
        return torch.zeros(1, self.hidden_size)
