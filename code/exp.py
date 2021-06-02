import importlib, random, argparse, time
import json
import numpy as np
import torch
from torch.utils.data import DataLoader

from model import Model
from trainer import Trainer
from utils import data_generator, SeqDataset

################################################################################
# Settings
################################################################################
parser = argparse.ArgumentParser()

parser.add_argument('--seed', type=int, default=999, help='manual random seed')
parser.add_argument('--continue_train', action='store_true')
parser.add_argument('--new_data', action='store_true')
parser.add_argument('--is_test', action='store_true')


parser.add_argument('--input_dim', type=int, default=64)
parser.add_argument('--output_dim', type=int, default=64)
parser.add_argument('--n_layer', type=int, default=2)

parser.add_argument('--ksize', type=int, default=7)
parser.add_argument('--rnn_type', type=str, default='gru')
parser.add_argument('--dropout', type=float, default=0.01)
parser.add_argument('--weight_decay', type=float, default=1e-6)


parser.add_argument('--n_epochs', type=int, default=100)
parser.add_argument('--batch_size', type=int, default=64)
parser.add_argument('--lr', type=float, default=0.001)
parser.add_argument('--n_gpu', type=int, default=1, help='number of GPU')


parser.add_argument('--data', type=str, default="log")

parser.add_argument('--alpha', type=float, default=1.0)

args = parser.parse_args()


class Logger(object):
  def __init__(self, filename):
    self.file = open(filename, 'a')

  def info(self, string):
    print (string)
    self.file.write(string + '\n')

  def done(self):
    self.file.close()

def main():

    model_name = "alpha_{}_lr_{}_id_{}_od_{}_ne_{}_{}_nl_{}_ks_{}".format(
                                            args.alpha, args.lr, args.input_dim, args.output_dim, 
                                            args.n_epochs, args.rnn_type, args.n_layer, args.ksize)

    log_filename = "./result/{}_result/{}.txt".format(args.data, model_name)
    model_filename = "./result/{}_models/{}.pt".format(args.data, model_name)
    result_filename = "./result/{}_result/{}.json".format(args.data, model_name)
    logger = Logger(log_filename)

    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)


    device = torch.device("cuda:0" if (torch.cuda.is_available() and args.n_gpu > 0) else "cpu")
    logger.info('Computation device: %s' % device)
    logger.info('Learning rate: {}, Batch_size: {}, # epochs: {}'.format(args.lr,args.batch_size, args.n_epochs))
    logger.info("Emb dim: {}, Ouput dim: {}, type: {}, nlayers: {}".format(args.input_dim,
                                                    args.output_dim, args.rnn_type, args.n_layer))
    
    #######################################################################
    ######                           DATA                           #######
    #######################################################################


    start_time = time.time()
    dataroot = "../datasets/{}".format(args.data)
    logger.info('Loading data from {}'.format(dataroot))
    corpus = data_generator(dataroot, redo=args.new_data)


    logger.info('num_token {} max len {}'.format(corpus.n_token, corpus.max_len))
    logger.info('Training data shape: {} x {}'.format(*corpus.train_x.shape))

    logger.info('Model Name: {}'.format(model_name))
    vocab_size = corpus.n_token
    train_data = SeqDataset(corpus.train_x, corpus.train_y, corpus.train_l_list)
    valid_data = SeqDataset(corpus.valid_x, corpus.valid_y, corpus.valid_l_list)
    test_data =  SeqDataset(corpus.test_x, corpus.test_y, corpus.test_l_list)

    workers = 1
    train_loader = DataLoader(train_data, batch_size=args.batch_size,
                                shuffle=True, num_workers=workers)
    valid_loader = DataLoader(valid_data, batch_size=args.batch_size,
                                shuffle=False, num_workers=workers)
    test_loader = DataLoader(test_data, batch_size=args.batch_size,
                                shuffle=False, num_workers=workers)

    logger.info('Data time: {:.3f}'.format(time.time() - start_time))

    #######################################################################
    ######                           Model                           ######
    #######################################################################


    net = Model(vocab_size, args.input_dim, args.output_dim, 
                          args.n_layer, args.rnn_type, args.ksize, args.dropout, device).to(device)


    if not args.is_test:
        trainer = Trainer(alpha=args.alpha, gc=None, lc=None, lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size,
                    weight_decay=args.weight_decay, device=device)

        # Train model on dataset
        net = trainer.train(train_loader, net, logger)
        torch.save({'alpha': trainer.alpha, 'gc': trainer.gc, 'lc': trainer.lc, 'net_dict': net.state_dict()}, model_filename)
    
    else:
        checkpoint = torch.load(model_filename)
        trainer = Trainer(checkpoint['alpha'], checkpoint['gc'], checkpoint['lc'], 
                            lr=args.lr, n_epochs=args.n_epochs, batch_size=args.batch_size,
                            weight_decay=args.weight_decay, device=device)
        net.load_state_dict(checkpoint['net_dict'])

    logger.info('Model name: {}'.format(model_name))

    results = {}

    trainer.test(valid_loader, net, logger)
    results['valid_auc'] = trainer.test_auc
    results['valid_time'] = trainer.test_time
    results['valid_scores'] = trainer.test_scores

    trainer.test(test_loader, net, logger)
    results['test_auc'] = trainer.test_auc
    results['test_time'] = trainer.test_time
    results['test_scores'] = trainer.test_scores

    with open(result_filename, 'w') as fp:
            json.dump(results, fp)


if __name__ == '__main__':
    main()
