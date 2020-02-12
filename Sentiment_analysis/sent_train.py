# coding=utf-8
import torch
import torch.nn as nn
import torch.optim as optim
import torch.optim.lr_scheduler as ReduceLROnPlateau
from torchtext import data
import numpy as np
import tqdm
from typing import Tuple

tokenize = lambda x: x.split()

""" Define the model """


class LSTMModel(nn.Module):

    def __init__(self, vocab, vocab_size, output_dim, emb_dim=200, hidden_dim=512, hidden_dim_2=512,
                 device=torch.device("cpu"), layers=2):

        super(LSTMModel, self).__init__()

        self.vocab_size = vocab_size
        self.emb_dim = emb_dim
        self.hidden_dim = hidden_dim
        self.hidden_dim_2 = hidden_dim_2

        # embedding layer
        self.embedding = nn.Embedding(vocab_size, emb_dim)
        self.embedding.weight.data.copy_(vocab.vectors)
        # lstm layer
        self.lstm = nn.LSTM(emb_dim, hidden_dim*2, num_layers=layers, bidirectional=True)
        # dropout layer
        self.dropout_layer_1 = nn.Dropout(p=0.5)
        # batch normalisation
        self.bn = nn.BatchNorm1d(hidden_dim*2)
        # linear layer
        self.hidden2hidden = nn.Linear(hidden_dim*2, hidden_dim_2)
        # dropout layer
        self.dropout_layer_2 = nn.Dropout(p=0.2)
        # relu layer
        self.relu = nn.ReLU()
        # linear layer
        self.hidden2out = nn.Linear(hidden_dim_2, output_dim)

        self.device = device

    def init_hidden(self, batch_size):
        # initialisation of hidden states
        return (torch.autograd.Variable(torch.nn.init.xavier_normal_(torch.Tensor(4, batch_size, self.hidden_dim))).to(
            device=self.device),
                torch.autograd.Variable(torch.nn.init.xavier_normal_(torch.Tensor(4, batch_size, self.hidden_dim))).to(
                    device=self.device))

    def forward(self, batch):

        self.hidden = self.init_hidden(batch.size(-1))
        embeddings = self.dropout_layer_2(self.embedding(batch))
        outputs, (ht, ct) = self.lstm(embeddings)

        # ht is last hidden state of sequences; ht = (1 x batch_size x hidden_size)
        # ht[-1] = (batch_size x hidden_size)
        feature = ht[-1]
        output = self.dropout_layer_1(feature)
        output = self.hidden2out(self.relu(self.dropout_layer_2(self.hidden2hidden(output))))
        return output


""" Define the generator """


class BatchGenerator:
    def __init__(self, iterator, x_field, y_field):
        self.iterator, self.x_field, self.y_field = iterator, x_field, y_field

    def __len__(self):
        return len(self.iterator)

    def __iter__(self):
        for batch in self.iterator:
            x = getattr(batch, self.x_field)
            y = getattr(batch, self.y_field)
            yield (x, y)


class SentimentAnalysis:

    def __init__(self, train_file, dev_file, test_file, seed=42):
        # Set random seed
        np.random.seed(seed)
        torch.manual_seed(seed)

        self.train_file = train_file
        self.dev_file = dev_file
        self.test_file = test_file

        self.device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")

        # define the fields
        self.text = data.Field(sequential=True, tokenize=tokenize, lower=True,
                               stop_words=["!", ".", ",", ":", "?", "\"", "\'", "(", ")", ";", "`", "\n", "\\", "/"])
        self.class_label = data.Field(sequential=False, use_vocab=True, is_target=True, pad_token=None, unk_token=None)

        # build datasets
        datasets = self.build_dataset()

        # build vocabulary
        self.text.build_vocab(datasets[0], vectors="glove.twitter.27B.200d")       # self.text.build_vocab(datasets[0], min_freq = 2)
        self.class_label.build_vocab(datasets[0])

        self.vocab_size = len(self.text.vocab)      # length of vocabulary

        self.classes = len(self.class_label.vocab)     # number of classes

        # build and wrap the iterators
        tr_iter, d_iter, tst_iter = self.build_iterator(datasets)
        self.train_iter = BatchGenerator(tr_iter, 'text', 'class_label')
        self.dev_iter = BatchGenerator(d_iter, 'text', 'class_label')
        self.test_iter = BatchGenerator(tst_iter, 'text', 'class_label')

        # define the model
        self.model = LSTMModel(self.text.vocab, self.vocab_size, self.classes)
        self.model = self.model.to(device=self.device)

    def build_dataset(self) -> Tuple[data.Dataset]:
        """
        Take files with training data, development data, test data as input
        :return: training, development, test datasets
        """
        datasets = data.TabularDataset.splits(path='', train=self.train_file, validation=self.dev_file,
                                              test=self.test_file, format='tsv', skip_header=False,
                                              fields=[("class_label", self.class_label), ("text", self.text)])
        return datasets

    def build_iterator(self, datasets, train_batch_size=32, dev_batch_size=256, test_batch_size=256):
        """ Build iterator for training, development and test datasets """
        return data.BucketIterator.splits(
            datasets, batch_sizes=(train_batch_size, dev_batch_size, test_batch_size),
            sort_key=lambda x: len(x.text), device=self.device, sort_within_batch=True, repeat=False)

    def evaluate(self, iter):
        """ Compute the accuracy for development and test data """
        self.model.eval()

        with torch.no_grad():
            total = 0
            correct = 0
            for x, y in iter:
                preds = self.model(x)
                for gold_label, predicted_label in zip(y, preds):
                    total += 1
                    if gold_label.item() == np.argmax(predicted_label):
                        correct += 1
        return correct / total

    def prediction(self, data):
        """ Compute the prediction for input """
        for x in data:
            preds = self.model(x)
            preds = preds.data.cpu().numpy()
        return preds

    def train(self, lr=0.1, epochs=500):
        """ Train model on training dataset and validate on development dataset"""

        opt = optim.SGD(self.model.parameters(), lr=lr)
        loss_func = nn.CrossEntropyLoss()

        # print model summary
        print(self.model)

        """
        Use ReduceLROnPlateau for earlier training stopping. 
        It reduces learning rate when a metric has stopped improving. This scheduler reads a metrics quantity and 
        if no improvement is seen for a ‘patience’ number of epochs, the learning rate is reduced.
        =============
        Parameters:
        =============
        factor = a factor by which the learning rate will be reduced. new_lr = lr * factor
        patience = number of epochs with no improvement after which learning rate will be reduced
        mode = min/max. In max mode learning rate will be reduced when the quantity monitored has stopped increasing. 
        min_lr = a lower bound on the learning rate of all param groups or each group respectively.
        """
        factor = 0.5
        patience = 3
        mode = "max"
        min_lr = 0.0001
        best_dev_acc = 0.0

        # scheduler: ReduceLROnPlateau = ReduceLROnPlateau.ReduceLROnPlateau(
        #     opt, factor=factor, patience=patience, mode=mode, verbose=True)

        print("Start training....")

        for epoch in range(epochs):

            train_loss = 0.0
            train_iterations = 0

            self.model.train()  # turn on training mode

            prev_lr = lr

            # for scheduler
            # for group in opt.param_groups:
            #     lr = group["lr"]

            # when the learning rate is lower than the threshold (min_lr), the training is stopped
            # if lr < min_lr:
            #     print("Learning rate is too small, the training is stopped.")
            #     break

            for x, y in tqdm.tqdm(self.train_iter):
                train_iterations += 1
                opt.zero_grad()
                preds = self.model(x)

                loss = loss_func(preds, y)
                loss.backward()

                opt.step()
                train_loss += loss.item()

            # calculate train loss
            train_loss /= train_iterations

            # validation on development data

            dev_loss = 0.0
            dev_iterations = 0
            self.model.eval()  # turn on evaluation mode

            with torch.no_grad():   # deactivate the autograd engine; reduce memory usage; speed up computations
                for x, y in tqdm.tqdm(self.dev_iter):
                    dev_iterations += 1
                    preds = self.model(x)
                    loss = loss_func(preds, y)
                    dev_loss += loss.item()

            # scheduler.step(dev_loss)

            # calculate the development accuracy
            dev_acc = self.evaluate(self.dev_iter)
            # calculate the development loss
            dev_loss /= dev_iterations

            print('Epoch: {}, Training Loss: {:.4f}, Validation Loss: {:.4f}, Validation Accuracy: {:.4f}, '
                  'Learning rate: {:.4f}'.format(epoch, train_loss, dev_loss, dev_acc, prev_lr))

            # save the trained model, if the accuracy is better than before
            if dev_acc > best_dev_acc:
                torch.save(self.model.state_dict(), "trained_model.pt")
                print("Trained model is saved in trained_model.pt")
                best_dev_acc = dev_acc

    def test(self, model):
        """ Test trained model on test dataset"""
        self.model.load_state_dict(torch.load(model))           # load the trained model
        test_acc = self.evaluate(self.test_iter)        # evaluate on the test set
        print("Test Accuracy of trained model: " + str(test_acc))


if __name__ == '__main__':
    analysis = SentimentAnalysis(train_file='data/sentiment.train.txt',
                                 dev_file='data/sentiment.dev.txt',
                                 test_file='data/sentiment.test.txt')

    analysis.train()
    analysis.test("trained_model.pt")
