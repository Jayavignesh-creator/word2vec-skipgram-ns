# Word2Vec (NumPy) - Quick Start

This repo contains a small Word2Vec skip-gram + negative sampling training pipeline written in NumPy. It uses wikipedia's text8 dataset for training.

## Project Structure

- `src/config.py`: training and data settings (`ModelConfig`)
- `src/data.py`: text loading, vocabulary creation, negative-sampling distribution, pair generation
- `src/model.py`: Word2Vec model, training loop, nearest-neighbor lookup
- `src/trainer.py`: end-to-end training entry point
- `src/data/text8`: training corpus used by default

## Requirements

- Python 3.9+ (recommended)
- `numpy`

Install dependency:

```bash
pip install numpy
```

## How To Run

From repo root:

```bash
python -m src.trainer
```

You can also run:

```bash
python src/trainer.py
```

Expected flow:
1. Loads tokens from `src/data/text8`
2. Builds capped vocabulary
3. Creates skip-gram `(center, context)` pairs
4. Trains embeddings
5. Prints nearest words for `b"anarchism"`

## Configure Training

All settings live in `src/config.py` under `ModelConfig`.

```python
class ModelConfig:
    data_path = "src/data/text8"
    vocab_size = 30000
    embedding_dim = 100
    window = 5
    neg_prob_power = 0.75
    negative_samples = 5
    lr = 0.025
    epochs = 1
    load_tokens = 800_000
    train_ids = 200_000
```

### Config Variables

| Variable | What it controls | Typical effect |
|---|---|---|
| `data_path` | Path to training corpus | Change dataset source |
| `vocab_size` | Max number of most frequent tokens to keep | Higher = richer vocab, more memory/time |
| `embedding_dim` | Embedding vector size | Higher = more capacity, slower training |
| `window` | Context window radius in pair generation | Higher = more pairs, broader context |
| `neg_prob_power` | Exponent for negative-sampling distribution | `0.75` is standard Word2Vec choice |
| `negative_samples` | Number of negatives per positive pair | Higher = stronger contrast, slower |
| `lr` | Learning rate for SGD updates | Too high can diverge, too low learns slowly |
| `epochs` | Full passes over generated pairs | More epochs usually improve fit up to a point |
| `load_tokens` | Number of tokens read from corpus | Controls dataset size and runtime |
| `train_ids` | Number of token IDs used to create pairs | Controls training subset size |

## Practical Tuning Tips

- For faster experiments:
  - reduce `load_tokens`, `train_ids`, `embedding_dim`, and `negative_samples`
- For potentially better quality:
  - increase `epochs` first, then `embedding_dim` and `train_ids`
- If training is unstable:
  - lower `lr` (for example from `0.025` to `0.01`)

## Notes

- The corpus is loaded in binary mode, so tokens are handled as `bytes` (not Python `str`).
- Nearest-neighbor queries should use bytes keys, e.g. `b"anarchism"`.
# word2vec-skipgram-ns
