import torch
import numpy as np
from torchtext import data
from Sentiment_analysis.sent_train import LSTMModel
import sys


class BatchGenerator:
    def __init__(self, iterator, x_field):
        self.iterator, self.x_field = iterator, x_field

    def __iter__(self):
        for batch in self.iterator:
            x = getattr(batch, self.x_field)
            yield x


def predict(model_path, answers_path):

    tokenize = lambda x: x.split()
    text = data.Field(sequential=True, tokenize=tokenize, lower=True,
                      stop_words=["!", ".", ",", ":", "?", "\"", "\'", "(", ")", ";", "`", "\n", "\\", "/"])
    class_label = data.Field(sequential=False, use_vocab=True, is_target=True, pad_token=None, unk_token=None)

    # datasets = data.TabularDataset.splits(path='Sentiment_analysis/data/', train='sentiment.train.txt',
    #                                       test='answers.txt', format='tsv', fields=[("text", text_field)])
    test_dataset = data.TabularDataset(path=answers_path, format='tsv', fields=[("text", text)])
    train_dataset = data.TabularDataset(path='Sentiment_analysis/data/sentiment.train.txt', format='tsv',
                                        fields=[("class_label", class_label), ("text", text)], skip_header=False)

    # build vocabulary
    text.build_vocab(train_dataset, vectors="glove.twitter.27B.200d")       # min_freq=1
    vocab_size = len(text.vocab)

    # build interator & batch generator
    iter = data.Iterator(test_dataset, batch_size=256)
    test_iter = BatchGenerator(iter, 'text')

    # load model
    model = LSTMModel(text.vocab, vocab_size, 5)
    model.load_state_dict(torch.load(model_path))  # load the trained model

    # do predictions
    final_preds = []
    with torch.no_grad():
        for x in test_iter:
            preds = model(x)
            sent_pred = np.argmax(preds.numpy(), axis=1)
            final_preds += sent_pred.tolist()
        final = sum(final_preds) / len(final_preds) / 4
    return round(final, 2)


if __name__ == '__main__':
    predict("trained_model.pt", "answers.txt")

# 0 - sehr negative
# 4 - sehr positive
