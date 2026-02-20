import numpy as np
from collections import Counter

class DataLoader():
    def __init__(self, data_path: str, num_tokens: int, vocab_size: int,  neg_prob_power: float, window_size: int) -> None:
        """Store data and preprocessing settings used to build training inputs."""
        self.data_path = data_path
        self.num_tokens = num_tokens
        self.vocab_size = vocab_size
        self.neg_prob_power = neg_prob_power
        self.window_size = window_size
    
    def load_data(self):
        """Read the raw corpus as bytes and return the first `num_tokens` tokens."""
        with open(self.data_path, "rb") as f:
            text = f.read()
        tokens = text.split()[:self.num_tokens]
        return tokens

    def get_neg_probs(self, word_freqs: np.ndarray):
        """Build the negative-sampling distribution using the frequency power trick."""
        neg_probs = word_freqs ** self.neg_prob_power
        neg_probs /= neg_probs.sum()
        return neg_probs
    
    def create_vocabulary(self):
        """Create vocab/id mappings, filter tokens, and return token ids with neg probs."""
        tokens = self.load_data()
        token_counts = Counter(tokens)
        print("Unique words:", len(token_counts))
        
        common_words = token_counts.most_common(self.vocab_size)
        print("Unique words after capping:", len(common_words))

        vocab = [w for w, _ in common_words]
        freqs = np.array([c for _, c in common_words], dtype=np.float64)

        word2id = {w: i for i, w in enumerate(vocab)}
        id2word = {i: w for w, i in word2id.items()}

        tokens = [w for w in tokens if w in word2id]
        print("Tokens after filtering:", len(tokens))

        ids = np.array([word2id[w] for w in tokens], dtype=np.int32)
        neg_probs = self.get_neg_probs(freqs)

        print("Vocabulary created size : ", len(vocab))
        return ids, neg_probs, word2id, id2word

    def make_dataset(self, ids: np.ndarray):
        """Convert token ids into (center, context) skip-gram training pairs."""
        pairs = []
        window = self.window_size
        n = len(ids)
        for i, center in enumerate(ids):
            left = max(0, i-window)
            right = min(n, i+window+1)
            for j in range(left, right):
                if j!=i:
                    pairs.append((center, ids[j]))
        
        print("Training pairs created size : ", len(pairs))
        return pairs
    

        
