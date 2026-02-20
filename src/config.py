class ModelConfig:
    """Central place for training hyperparameters and dataset settings."""

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
