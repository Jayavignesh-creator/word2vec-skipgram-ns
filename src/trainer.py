from src.config import ModelConfig as mc
from src.data import DataLoader
from src.model import Word2Vec

class Modeltrainer:
    def __init__(self):
        """Set up containers for learned vocabulary mappings."""
        self.word2id = {}
        self.id2word = {}
    
    def train_word2vec(self):
        """Build data, train Word2Vec, and print nearest neighbors for a sample word."""
        data_loader = DataLoader(mc.data_path, mc.load_tokens, mc.vocab_size, mc.neg_prob_power, mc.window)
        ids, neg_probs, word2id, id2word = data_loader.create_vocabulary()

        ids_train = ids[:mc.train_ids]

        self.word2id = word2id
        self.id2word = id2word

        pairs = data_loader.make_dataset(ids_train)

        actual_vocab_size = len(word2id)
        model = Word2Vec(actual_vocab_size, mc.embedding_dim, word2id, id2word)
        model.init_weights()
        model.train_from_pairs(pairs, neg_probs, mc.epochs, mc.lr, mc.negative_samples)

        topk_nearest = model.get_topk(b"anarchism")
        print(topk_nearest)

if __name__ == "__main__":
    Modeltrainer().train_word2vec()
    
