# File with config parameters definitions that will be read in during runs


params = dict(
    model = "CM", # "CM", "MM"
    temperature = 10, # temperature of the gumbel softmax
    normalization = "gumbel", # Which normalization to apply to causal parameters. Choices "softmax", "gumbel", "none"
    hard = True, # Whether to use hard thresholding in gumbel/softmax
    num_runs = 50,
    num_epochs = 300,
    num_batches = 5,
    batch_size = 1024,
    eval_size = 5, # On how many datapoints to evaluate the model after each epoch
    distribution_change_freq = 1, # Number of epochs before the prior p(A) is changed. If this is > num_epochs there won't be a change in the distribution
    n_categories = 5, # Categories of each one-hot encoded data point
    use_bias=False, # Whether to add a bias term to Q, K and V
    emb_size=10, # The embedding size of matrices Q, K, V
    positional_encoding = "none", # 'none', 'same', 'different'
    pos_enc_type = "uniform", # 'gauss' or 'ramp', 'uniform' for positional_encoding="same"
    pos_enc_scale=1, # std for pos_enc_type='gauss' and min/max for pos_enc_type='ramp'
    lr=0.1, # learning rate,
    use_input=True, # Whether to provide A,B input to the model,
    record_video=False, # Whether to record a video visualization of the matrices
    factorization="b->a", #"a->b", "b->a", "a,b"
    smoothness=1,
    format="one-hot",#"one-hot", "noise", "digit"
    loss="marginal",#"marginal", "joint"
    beta=1,
)
