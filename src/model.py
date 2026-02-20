import numpy as np

class Word2Vec():
    def __init__(self, vocab_size: int, embedding_dim: int, word2id: dict, id2word: dict) -> None:
        """Initialize model dimensions, vocab mappings, and random state."""
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.word2id = word2id
        self.id2word = id2word
        self.rng = np.random.default_rng(42)
        self.W = None
        self.WT = None

    def init_weights(self):
        """Create input/output embedding matrices with small random values."""
        self.W = (self.rng.random((self.vocab_size, self.embedding_dim)) - 0.5) / self.embedding_dim
        self.WT = self.rng.random((self.vocab_size, self.embedding_dim))

        print("Weights Initialized. W_in: ", self.W.shape, "W_out: ", self.WT.shape)

    def sigmoid(self, x):
        """Apply a clipped sigmoid for safer numerical behavior."""
        x = np.clip(x, -15, 15)
        return 1 / (1 + np.exp(-x))

    def train_step(self, center_id, pos_id, neg_probs, K=5, lr=0.025):
        """Run one skip-gram negative-sampling update and return step loss."""
        v = self.W[center_id]

        u_pos = self.WT[pos_id]

        neg_ids = self.rng.choice(self.vocab_size, size=K, p=neg_probs)
        u_neg = self.WT[neg_ids]

        s_pos = u_pos @ v
        s_neg = u_neg @ v

        loss = -np.log(self.sigmoid(s_pos) + 1e-10) - np.sum(np.log(self.sigmoid(-s_neg) + 1e-10))

        g_pos = self.sigmoid(s_pos) - 1.0
        g_neg = self.sigmoid(s_neg)

        grad_v = g_pos * u_pos + (g_neg[:, None] * u_neg).sum(axis=0)
        grad_u_pos = g_pos * v
        grad_u_neg = g_neg[:, None] * v[None, :]

        self.W[center_id] -= lr * grad_v
        self.WT[pos_id]   -= lr * grad_u_pos

        for i, nid in enumerate(neg_ids):
            self.WT[nid] -= lr * grad_u_neg[i]

        return loss

    def train_from_pairs(self, pairs, neg_probs, epochs=1, lr=0.025, K=5, log_every=200000):
        """Train the model for multiple epochs over precomputed training pairs."""
        pairs = np.array(pairs, dtype=np.int32)

        step = 0
        for ep in range(1, epochs + 1):
            self.rng.shuffle(pairs)

            total_loss = 0.0
            epoch_step = 0
            for center_id, pos_id in pairs:
                loss = self.train_step(center_id, pos_id, neg_probs, K=K, lr=lr)
                total_loss += loss
                epoch_step += 1

                step += 1
                if log_every and step % log_every == 0:
                    print(f"epoch {ep} step {step} avg_loss={total_loss/epoch_step:.4f}")

            print(f"epoch {ep} done | avg_loss={total_loss/len(pairs):.4f}")

    def get_topk(self, word, topk=10):
        """Return the top-k nearest words by cosine similarity in input embeddings."""
        if word not in self.word2id:
            return []
        i = self.word2id[word]
        w = self.W[i]
        sims = (self.W @ w) / (np.linalg.norm(self.W, axis=1) * np.linalg.norm(w) + 1e-10)
        best = np.argsort(-sims)[:topk+1]
        return [(self.id2word[j], float(sims[j])) for j in best if j != i][:topk]
