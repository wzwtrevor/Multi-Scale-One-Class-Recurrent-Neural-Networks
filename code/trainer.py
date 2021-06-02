import time
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
import torch.optim as optim
import torch

class Trainer(object):

    def __init__(self, alpha, gc, lc, lr, n_epochs, batch_size, weight_decay, device):


        # training configuration
        self.alpha = alpha
        self.lr = lr
        self.n_epochs = n_epochs
        self.batch_size = batch_size
        self.weight_decay = weight_decay
        self.device = device

        # model related
        self.gc = torch.tensor(gc, device=self.device) if gc is not None else None
        self.lc = torch.tensor(lc, device=self.device) if lc is not None else None

        # Results
        self.train_time = None
        self.test_auc = None
        self.test_time = None
        self.test_scores = None

    def train(self, train_loader, net, logger):

        net = net.to(self.device)

        optimizer = optim.Adam(net.parameters(), lr=self.lr, weight_decay=self.weight_decay)

        if self.gc is None:
            logger.info('Initializing center ...')
            self.gc, self.lc = self.init_center_c(train_loader, net)
            logger.info('Center initialized.')
  

        logger.info('Starting training...')
        start_time = time.time()
        net.train()
        for epoch in range(self.n_epochs):

            loss_epoch = 0.0
            gloss_epoch = 0.0
            lloss_epoch = 0.0

            n_batches = 0
            epoch_start_time = time.time()


        
            for i, (inputs, label, l_list) in enumerate(train_loader):
               
                inputs = inputs.to(self.device)
                l_list = l_list.to(self.device)

                # Zero the network parameter gradients
                optimizer.zero_grad()

                g_outputs, l_outputs = net(inputs, l_list)
                g_dist = torch.sum((g_outputs - self.gc) ** 2, dim=1)
        

                l_all_dist = torch.sum((l_outputs - self.lc) ** 2, dim=2)
                l_dist = torch.max(l_all_dist, dim=1)[0]

                dist = g_dist + self.alpha*l_dist

                loss = torch.mean(dist)
                loss.backward()
                optimizer.step()

                gloss_epoch += torch.mean(g_dist)
                lloss_epoch += torch.mean(l_dist)
                loss_epoch += loss.item()
                n_batches += 1

            epoch_train_time = time.time() - epoch_start_time
            if epoch % 1 == 0:
                logger.info('  Epoch {}/{}\t Time: {:.3f}\t Global Loss: {:.8f}\t Local Loss {:.8f}'
                            .format(epoch + 1, self.n_epochs, epoch_train_time, gloss_epoch / n_batches, lloss_epoch / n_batches))

        self.train_time = time.time() - start_time
        logger.info('Training time: %.3f' % self.train_time)

        logger.info('Finished training.')

        return net

    def test(self, test_loader, net, logger):

        net = net.to(self.device)

        logger.info('Starting testing...')
        start_time = time.time()
        idx_label_score = []
        net.eval()
        with torch.no_grad():
            for i, (inputs, labels, l_list) in enumerate(test_loader):

                    
                inputs = inputs.to(self.device)
                l_list = l_list.to(self.device)
                g_outputs, l_outputs = net(inputs, l_list)

                g_dist = torch.sum((g_outputs - self.gc) ** 2, dim=1)

                l_all_dist = torch.sum((l_outputs - self.lc) ** 2, dim=2)
                l_dist = torch.max(l_all_dist, dim=1)[0] 

                dist = g_dist + self.alpha*l_dist
                scores = dist

                idx_label_score += list(zip(labels.cpu().data.numpy().tolist(),
                                             scores.cpu().data.numpy().tolist()))

        self.test_time = time.time() - start_time
        logger.info('Testing time: %.3f' % self.test_time)

        self.test_scores = idx_label_score

        labels, scores = zip(*idx_label_score)
        labels = np.array(labels)
        scores = np.array(scores)

        self.test_auc = roc_auc_score(labels, scores)
        self.average_p = average_precision_score(labels, scores)
        logger.info('Test set AUC: {:.2f} AP: {:.2f}'.format(100. * self.test_auc, 100.*self.average_p))

        logger.info('Finished testing.')

    def init_center_c(self, train_loader, net, eps=0.1):
        """Initialize hypersphere center c as the mean from an initial forward pass on the data."""
        n_samples = 0
        gc = torch.zeros(net.output_dim, device=self.device)
        lc = torch.zeros(net.output_dim, device=self.device)

        net.eval()
        with torch.no_grad():
            for inputs, labels, l_list in train_loader:
                # get the inputs of the batch
                inputs = inputs.to(self.device)
                l_list = l_list.to(self.device)
                g_outputs, l_outputs = net(inputs, l_list)
                n_samples += g_outputs.shape[0]
                gc += torch.sum(g_outputs, dim=0)
                lc += torch.sum(l_outputs.mean(dim=1), dim=0)
        gc /= n_samples
        lc /= n_samples

        gc[(abs(gc) < eps) & (gc < 0)] = -eps
        gc[(abs(gc) < eps) & (gc > 0)] = eps


        lc[(abs(lc) < eps) & (lc < 0)] = -eps
        lc[(abs(lc) < eps) & (lc > 0)] = eps

        return gc, lc

