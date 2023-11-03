import argparse
import scipy.sparse as sp
from gensim.models import Word2Vec
from sklearn.svm import SVC
from sklearn.metrics import classification_report
from sklearn.model_selection import train_test_split
from walker import BiasedRandomWalker
from utils import *


class trans2vec(object):
    def __init__(self, alpha=0.5, dimensions=64, num_walks=20, walk_length=5, window_size=10, workers=1, seed=2022):
        self.alpha = alpha
        self.dimensions = dimensions
        self.window_size = window_size
        self.workers = workers
        self.seed = seed
        self.walk_length = walk_length
        self.num_walks = num_walks

        self.walks = None
        self.word2vec_model = None
        self.embeddings = None
        self.do()

    def do(self):
        self.load_data()
        self.walk()

    def load_data(self):
        """load data from the npz processed by Lin
            refer to <https://arxiv.org/abs/1905.08038>
        """
        data = np.load('dataset/tedge.npz', allow_pickle=True)
        self.adj_matrix = data['adj_matrix'].item()
        self.amount_data = data['amount_data'].item()
        self.timestamp_data = data['timestamp_data'].item()
        self.node_label = data['node_label']
        self.adj_matrix.data = self.get_amount_timestamp_data()

    def get_amount_timestamp_data(self):
        """Preprocessing transition probability: alpha * TBS * (1-alpha) * WBS
            refer to <https://ieeexplore.ieee.org/document/9184813>

            Returns
            -------
            amount_timestamp_data.data : sp.csr_matrix.data
        """
        N = self.adj_matrix.shape[0]
        amount_timestamp_data = sp.lil_matrix((N, N), dtype=np.float64)
        nodes = np.arange(N, dtype=np.int32)
        indices = self.adj_matrix.indices
        indptr = self.adj_matrix.indptr
        amount_data = self.amount_data.data
        timestamp_data = self.timestamp_data.data
        for node in nodes:
            nbrs = indices[indptr[node]: indptr[node + 1]]
            nbrs_amount_probs = amount_data[indptr[node]: indptr[node + 1]].copy()
            nbrs_timestamp_probs = timestamp_data[indptr[node]: indptr[node + 1]].copy()
            nbrs_unnormalized_probs = combine_probs(nbrs_amount_probs, nbrs_timestamp_probs, self.alpha)

            for i, nbr in enumerate(nbrs):
                amount_timestamp_data[node, nbr] = nbrs_unnormalized_probs[i]
        amount_timestamp_data = amount_timestamp_data.tocsr()
        return amount_timestamp_data.data

    def walk(self):
        walks = BiasedRandomWalker(walk_length=self.walk_length, walk_number=self.num_walks).walk(self.adj_matrix)
        word2vec_model = Word2Vec(sentences=walks, vector_size=self.dimensions, window=self.window_size,
                                  min_count=0, sg=1, hs=1, workers=self.workers, seed=self.seed)
        embeddings = word2vec_model.wv.vectors[np.fromiter(map(int, word2vec_model.wv.index_to_key), np.int32).argsort()]
        self.walks = walks
        self.word2vec_model = word2vec_model
        self.embeddings = embeddings


def node_classification(args, embeddings):
    labels_dict = load_labels('dataset/label.txt')
    nodes = list([int(node) for node in labels_dict.keys()])
    nodes_labels = list(labels_dict.values())
    nodes_embeddings = embeddings[nodes]

    X_train, X_test, y_train, y_test = train_test_split(nodes_embeddings, nodes_labels, train_size=args.train_size, random_state=args.seed)
    model = SVC(kernel='linear', C=0.4, random_state=args.seed)
    model.fit(X_train, y_train)
    y_pred = model.predict(X_test)
    cr = classification_report(y_pred, y_test)
    print('classification_report:\n{}'.format(cr))


def run_trans2vec(args):
    t2v = trans2vec(alpha=args.alpha, dimensions=args.dimensions, num_walks=args.num_walks,
                    walk_length=args.walk_length, window_size=args.window_size, workers=args.workers, seed=args.seed)
    embeddings = t2v.embeddings
    node_classification(args, embeddings)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", default=2022, type=int, help="random seed")
    parser.add_argument("-d", "--dimensions", default=64, type=int)
    parser.add_argument("--num_walks", default=5, type=int)
    parser.add_argument("--walk_length", default=2, type=int)
    parser.add_argument("--window_size", default=2, type=int)
    parser.add_argument("--workers", default=1, type=int)
    parser.add_argument("--train_size", default=0.8, type=float)
    parser.add_argument("--alpha", default=0.5, type=float, help="balance between TBS and WBS")
    args = parser.parse_args()

    random_seed(args.seed)
    run_trans2vec(args)

