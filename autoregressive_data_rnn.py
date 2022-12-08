import wget, os, gzip, pickle, random, re, sys
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
import torch.distributions as dist

IMDB_URL = 'http://dlvu.github.io/data/imdb.{}.pkl.gz'
IMDB_FILE = 'imdb.{}.pkl.gz'

PAD, START, END, UNK = '.pad', '.start', '.end', '.unk'

def load_imdb(final=False, val=5000, seed=0, voc=None, char=False):

    cst = 'char' if char else 'word'

    imdb_url = IMDB_URL.format(cst)
    imdb_file = IMDB_FILE.format(cst)

    if not os.path.exists(imdb_file):
        wget.download(imdb_url)

    with gzip.open(imdb_file) as file:
        sequences, labels, i2w, w2i = pickle.load(file)

    if voc is not None and voc < len(i2w):
        nw_sequences = {}

        i2w = i2w[:voc]
        w2i = {w: i for i, w in enumerate(i2w)}

        mx, unk = voc, w2i['.unk']
        for key, seqs in sequences.items():
            nw_sequences[key] = []
            for seq in seqs:
                seq = [s if s < mx else unk for s in seq]
                nw_sequences[key].append(seq)

        sequences = nw_sequences

    if final:
        return (sequences['train'], labels['train']), (sequences['test'], labels['test']), (i2w, w2i), 2

    # Make a validation split
    random.seed(seed)

    x_train, y_train = [], []
    x_val, y_val = [], []

    val_ind = set( random.sample(range(len(sequences['train'])), k=val) )
    for i, (s, l) in enumerate(zip(sequences['train'], labels['train'])):
        if i in val_ind:
            x_val.append(s)
            y_val.append(l)
        else:
            x_train.append(s)
            y_train.append(l)

    return (x_train, y_train), \
           (x_val, y_val), \
           (i2w, w2i), 2


def gen_sentence(sent, g):

    symb = '_[a-z]*'

    while True:

        match = re.search(symb, sent)
        if match is None:
            return sent

        s = match.span()
        sent = sent[:s[0]] + random.choice(g[sent[s[0]:s[1]]]) + sent[s[1]:]

def gen_dyck(p):
    open = 1
    sent = '('
    while open > 0:
        if random.random() < p:
            sent += '('
            open += 1
        else:
            sent += ')'
            open -= 1

    return sent

def gen_ndfa(p):

    word = random.choice(['abc!', 'uvw!', 'klm!'])

    s = ''
    while True:
        if random.random() < p:
            return 's' + s + 's'
        else:
            s+= word

def load_brackets(n=50_000, seed=0):
    return load_toy(n, char=True, seed=seed, name='dyck')

def load_ndfa(n=50_000, seed=0):
    return load_toy(n, char=True, seed=seed, name='ndfa')

def load_toy(n=50_000, char=True, seed=0, name='lang'):

    random.seed(0)

    if name == 'lang':
        sent = '_s'

        toy = {
            '_s': ['_s _adv', '_np _vp', '_np _vp _prep _np', '_np _vp ( _prep _np )', '_np _vp _con _s' , '_np _vp ( _con _s )'],
            '_adv': ['briefly', 'quickly', 'impatiently'],
            '_np': ['a _noun', 'the _noun', 'a _adj _noun', 'the _adj _noun'],
            '_prep': ['on', 'with', 'to'],
            '_con' : ['while', 'but'],
            '_noun': ['mouse', 'bunny', 'cat', 'dog', 'man', 'woman', 'person'],
            '_vp': ['walked', 'walks', 'ran', 'runs', 'goes', 'went'],
            '_adj': ['short', 'quick', 'busy', 'nice', 'gorgeous']
        }

        sentences = [ gen_sentence(sent, toy) for _ in range(n)]
        sentences.sort(key=lambda s : len(s))

    elif name == 'dyck':

        sentences = [gen_dyck(7./16.) for _ in range(n)]
        sentences.sort(key=lambda s: len(s))

    elif name == 'ndfa':

        sentences = [gen_ndfa(1./4.) for _ in range(n)]
        sentences.sort(key=lambda s: len(s))

    else:
        raise Exception(name)

    tokens = set()
    for s in sentences:

        if char:
            for c in s:
                tokens.add(c)
        else:
            for w in s.split():
                tokens.add(w)

    i2t = [PAD, START, END, UNK] + list(tokens)
    t2i = {t:i for i, t in enumerate(i2t)}

    sequences = []
    for s in sentences:
        if char:
            tok = list(s)
        else:
            tok = s.split()
        sequences.append([t2i[t] for t in tok])

    return sequences, (i2t, t2i)

def print_sequence(x_train, i):
    print(f"Sequence #{str(i).rjust(6, ' ')}: {''.join([i2w[i] for i in x_train[i]])}")

class Net(nn.Module):
    def __init__(self,
                 vocab_size,
                 embedding_size=32,
                 hidden_size=16,
                 lstm_num_layers=2,
                 *args, **kwargs):

        super().__init__()

        self.vocab_size = vocab_size
        self.embedding_size = embedding_size
        self.lstm_hidden_size = hidden_size
        self.lstm_num_layers = lstm_num_layers

        self.embed = nn.Embedding(self.vocab_size,
                                   self.embedding_size)
        self.lstm = nn.LSTM(input_size=self.embedding_size,
                            hidden_size=self.hidden_size,
                            num_layers=self.lstm_num_layers,
                            batch_first=True)
        self.linear = nn.Linear(self.hidden_size, self.vocab_size)

        def forward(self, x):
            input = self.embedding(x)
            lstm_output, (hn, cn) = self.lstm(input)
            output = self.linear(lstm_output)
            return output

# TODO:
# prepend '.start' token, append '.end' token
# to make target tensor, remove first column of the input tensor and append a column of zeros
# use torch.utils.data.Dataset
class NdfaDataset(Dataset):
    def __init__(self, corpus, w2i, i2w):
        self.corpus = corpus
        self.w2i = w2i
        self.i2w = i2w

    def __len__(self):
        return len(self.corpus)

    def __getitem__(self, index):
        for seq in self.corpus:
            seq.insert(0, '.start')
            seq.append('.end')
            for i in range(len(seq)):
                # add (x, y) = (seq[:i], seq[1:i+1]) to the dataset
                # TODO; need to understand how PyTorch Datasets work
                pass
        return

def train(dataloader, model, loss_fn, optimizer):
    size = len(dataloader.dataset)
    model.train()
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        pred = model(X)
        loss = loss_fn(pred, y)

        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if batch % 100 == 0:
            loss, current = loss.item(), batch * len(X)
            print(f"loss: {loss:>7f}  [{current:>5d}/{size:>5d}]")

# TODO
# very first rough draft for now:
# use sample function from assignment
def predict(dataset, model, seq, temperature=1.0, max_length=20):
    """
    :param dataset: need i2w and w2i
    :param model: the model we sample from
    :param seq: the sequence of tokens we want to complete
    :param max_length: we stop if we reach an end token, or after max_length tokens
    :return: the generated sequence of tokens
    """
    model.eval()
    pred = []
    for i in range(0, max_length):
        x = torch.tensor([[dataset.w2i[i] for w in seq[i:]]])
        y = model.forward(x)
        last_token_logits = y[0][-1]
        j = sample(last_token_logits, temperature)
        pred.append(seq.dataset.i2w[j])
        if seq.dataset.i2w[j] == '.end':
            return pred
    return pred


def sample(lnprobs, temperature=1.0):
    """
     Sample an element from a categorical distribution
     :param lnprobs: Outcome logits
     :param temperature: Sampling temperature. 1.0 follows the given distribution, 0.0 returns the maximum probability element.
     :return: The index of the sampled element.
    """
    if temperature == 0.0:
             return lnprobs.argmax()

    p = F.softmax(lnprobs / temperature, dim=0)
    cd = dist.Categorical(p)
    return cd.sample()


if __name__ == '__main__':

    # Generate ndfa sequences
    n = 150000
    print("Generating ndfa data")
    x_train, (i2w, w2i) = load_ndfa(n)
    print(f'Dictionary:{w2i}')
    print(f'Word index:{i2w}')

    # Print some random sequences
    for i in np.random.randint(n, size=10):
        print_sequence(x_train, i)

    # Create network
    net = Net(vocab_size=len(w2i))
    device = torch.device('mps' if torch.has_mps else 'cpu')
    net.to(device)
    print(f"Using {device} device")

    print("Creating dataset")
    dataset = NdfaDataset(x_train, w2i, i2w)


    # TODO:
    # Batches:
    # batch_size = maximum number of tokens
    # make batches of similar length sequences
    # check out torch.nn.utils.rnn.pack_padded_sequence() or torch.nn.utils.rnn.pack_sequence()
    # to make the batches
    dataloader = DataLoader(dataset, batch_size=16)

    # Loss function:
    # check whether the loss function applies softmax or whether we need to do it manually
    # loss function = cross entropy loss at every point in time, read doc to figure out
    # how to shuffle dimensions properly
    criterion = nn.CrossEntropyLoss()

    # Optimizer:
    optimizer = optim.Adam(net.parameters(), lr=0.001)

    # Training loop
    print("Training")
    epochs = 5
    for t in range(epochs):
        print(f"Epoch {t + 1}\n-------------------------------")
        print("Training:")
        train(dataloader, net, criterion, optimizer)

        print("Predicting:")
        seq = ['.start', 'a', 'b']
        predict(dataset, net, seq, max_length=20)

    print("Done!")





