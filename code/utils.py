import os, random, pickle
import torch
from torch.utils.data import Dataset

class SeqDataset(Dataset):
    def __init__(self, data, label, l_list):
        self.data = data
        self.label = label
        self.l_list = l_list

    def __getitem__(self, index):
        return self.data[index], self.label[index], self.l_list[index]

    def __len__(self):
        return self.data.shape[0]



def data_generator(data_path, redo=False):
    if os.path.exists(data_path + "/corpus") and not redo:
        corpus = pickle.load(open(data_path + '/corpus', 'rb'))
    else:
        corpus = Corpus(data_path)
        pickle.dump(corpus, open(data_path + '/corpus', 'wb'))
    return corpus



class Dictionary(object):
    def __init__(self):
        self.word2idx = {'PAD':0}
        self.idx2word = {0:'PAD'}
        self.pad = 0
        self.idx = 1

    def add_word(self, word):
        if word not in self.word2idx:
            self.idx2word[self.idx] = word
            self.word2idx[word] = self.idx # pad, sos, end
            self.idx += 1
        return self.word2idx[word]

    def __len__(self):
        return len(self.idx2word.keys())



class Corpus(object):
    def __init__(self, path):
        self.pad_idx = 0
        self.dictionary = Dictionary()
        self.max_len = 0

        train_file = os.path.join(path, 'train.txt')
        valid_file = os.path.join(path, 'valid.txt')
        test_file = os.path.join(path, 'test.txt')
        

        train_seqs, self.train_l_list = self.tokenize(train_file)
        valid_seqs, self.valid_l_list = self.tokenize(valid_file)
        test_seqs, self.test_l_list = self.tokenize(test_file)

        # l_list constains the last index, instead of length, which is 1 more
        self.max_len = max(torch.max(self.train_l_list), torch.max(self.valid_l_list), torch.max(self.test_l_list)) + 1

        self.train_x, self.train_y= self.vectorize(train_seqs, 0)

        if path.split('/')[-1] == 'log':
            self.valid_x, self.valid_y = self.vectorize(valid_seqs, 3368)
            self.test_x, self.test_y = self.vectorize(test_seqs, 13470)

        if path.split('/')[-1] == 'nec':
            self.valid_x, self.valid_y = self.vectorize(valid_seqs, 215)
            self.test_x, self.test_y = self.vectorize(test_seqs, 785)

        if path.split('/')[-1] == 'bgl':
            self.valid_x, self.valid_y = self.vectorize(valid_seqs, 257)
            self.test_x, self.test_y = self.vectorize(test_seqs, 552)


        self.n_token = len(self.dictionary)
        

    def tokenize(self, in_file):
        output = []
        l_list = []
        with open(in_file, 'r') as f:
            for line in f:
                idx_list = []
                tokens = line.strip().split()
                l = len(tokens)
                l_list.append(l-1)
                for token in tokens:
                    idx_list.append(self.dictionary.add_word(token))
                output.append(idx_list)
        return output, torch.LongTensor(l_list)
        
    def vectorize(self, seqs, bad):
        """Tokenizes a text file."""

        n_seq = len(seqs)
        # Tokenize file content
        data = torch.zeros((n_seq, self.max_len), dtype=torch.long)
        label = torch.zeros(n_seq, dtype=torch.long)
        for i, word_ids in enumerate(seqs):
            if i < bad: 
                label[i] = 1 # anomaly
            else:
                label[i] = 0
            for j, word_id in enumerate(word_ids):
                data[i][j] = word_id
        return data, label
