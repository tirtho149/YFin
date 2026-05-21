"""Model families benchmarked by finbench:

* ``mlp``         — feed-forward MLP regressor / classifier
* ``classical``   — XGBoost, RandomForest, Ridge / LogisticRegression
* ``sequence``    — LSTM and Transformer encoders over look-back windows
* ``world_model`` — latent world model (VAE encoder -> latent transition -> head)
"""
