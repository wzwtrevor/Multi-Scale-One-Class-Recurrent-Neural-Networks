import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np




class LocalRNN(nn.Module):
    def __init__(self, input_dim, output_dim, rnn_type, ksize, device):
        super(LocalRNN, self).__init__()
        """
        LocalRNN structure
        """
        self.output_dim = output_dim
        self.ksize = ksize
        assert rnn_type in ['gru', 'lstm', 'rnn']
        if rnn_type == 'gru':
            self.rnn = nn.GRU(input_dim, output_dim, batch_first=True)
        elif rnn_type=='lstm':
            self.rnn = nn.LSTM(input_dim, output_dim, batch_first=True)
        else:
            self.rnn = nn.RNN(input_dim, output_dim, batch_first=True)

        # To speed up
        idx = [i for j in range(self.ksize-1,10000,1) for i in range(j-(self.ksize-1),j+1,1)]
        self.select_index = torch.LongTensor(idx).to(device)
        self.zeros = torch.zeros((self.ksize-1, input_dim)).to(device)

    def forward(self, x):
        nbatches, l, input_dim = x.shape
        x = self.get_K(x) 
        batch, l, ksize, input_dim = x.shape
        h = self.rnn(x.view(-1, self.ksize, input_dim))[0][:,-1,:]
        return h.view(batch, l, self.output_dim)


    def get_K(self, x):
        batch_size, l, _ = x.shape
        zeros = self.zeros.unsqueeze(0).repeat(batch_size, 1, 1)
        x = torch.cat((zeros, x), dim=1)
        key = torch.index_select(x, 1, self.select_index[:self.ksize*l])
        key = key.reshape(batch_size, l, self.ksize, -1)
        return key

class Model(nn.Module):
    def __init__(self, vocab_size, input_dim, output_dim, n_layer, rnn_type, ksize, dropout, device):
        super(Model, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.emb = torch.nn.Embedding(vocab_size, input_dim)
        self.ksize = ksize
        self.local_rnn = LocalRNN(input_dim, output_dim, rnn_type, ksize, device)


        assert rnn_type in ['gru', 'lstm', 'rnn']
        self.rnn_type = rnn_type
        if rnn_type == 'gru':
            self.global_rnn = nn.GRU(input_dim, output_dim, num_layers=n_layer, batch_first=True)
        elif rnn_type== 'lstm': 
            self.global_rnn = nn.LSTM(input_dim, output_dim, num_layers=n_layer, batch_first=True)
        else:
            self.global_rnn = nn.RNN(input_dim, output_dim, num_layers=n_layer, batch_first=True)

      

        self.dropout = nn.Dropout(dropout)

    def forward(self, x, l_list):
        """
        Output dimension
                global: batch, output_dim
                local_output: batch, l, output_dim
        """
        x = self.emb(x)
        n, l, d = x.shape

        local_output = self.local_rnn(x)
        if self.rnn_type=='lstm':
            global_output, (global_hn, c_n) = self.global_rnn(x)
        else:
            global_output, global_hn = self.global_rnn(x)
        idx = l_list.reshape(n,1,1).expand(n, 1, self.output_dim) 

        return torch.gather(global_output,1,idx).squeeze(1), local_output
    

