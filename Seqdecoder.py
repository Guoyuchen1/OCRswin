'''
    Encoder+Decoder+Attention is implemented from:
    https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html
'''
import torch
import torch.nn as nn
import torch.nn.functional as F



class Decoder(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(Decoder, self).__init__()
        self.hidden_size = hidden_size
        self.decoder = AttnDecoder(hidden_size, output_size, dropout_p, max_length)

    def forward(self, input, hidden, encoder_outputs):

        return self.decoder(input, hidden, encoder_outputs)

    def initHidden(self, batch_size):
        result = torch.autograd.Variable(torch.zeros(1, batch_size, self.hidden_size))
        return result

class AttnDecoder(nn.Module):

    def __init__(self, hidden_size, output_size, dropout_p=0.1, max_length=71):
        super(AttnDecoder, self).__init__()
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.dropout_p = dropout_p
        self.max_length = max_length

        self.embedding = nn.Embedding(self.output_size, self.hidden_size)
        self.attn = nn.Linear(self.hidden_size * 2, self.max_length)
        self.attn_combine = nn.Linear(self.hidden_size * 4, self.hidden_size)
        self.dropout = nn.Dropout(self.dropout_p)
        self.gru = nn.GRU(self.hidden_size, self.hidden_size)
        self.out = nn.Linear(self.hidden_size, self.output_size)


    def forward(self, input, hidden, encoder_outputs):
        embedded = self.embedding(input)
        embedded = self.dropout(embedded)

        attn_weights = F.softmax(self.attn(torch.cat((embedded, hidden[0]), 1)), dim=1)
        attn_applied = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs.permute(1, 0, 2))

        output = torch.cat((embedded, attn_applied.squeeze(1)), 1)

        output = self.attn_combine(output).unsqueeze(0)

        output = F.relu(output)
        output, hidden = self.gru(output, hidden)

        output = F.log_softmax(self.out(output[0]), dim=1)
        return output, hidden, attn_weights

    def initHidden(self):
        return torch.zeros(1, 1, self.hidden_size, device=device)

