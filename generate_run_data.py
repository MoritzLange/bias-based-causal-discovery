import os
import sys

# The following is needed to be able to import from parent directory
module_path = os.path.abspath(os.path.join('..'))
if module_path not in sys.path:
    sys.path.append(module_path)

from utils.misc import evaluate, write_parameters, save_data
from torch.utils.data import Dataset
from scipy.special import softmax
from scipy.linalg import norm
from pathlib import Path

from tqdm import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import numpy.random as npr
from datetime import datetime
from types import SimpleNamespace

import pandas as pd

import torch
from torch import nn
import torch.nn.functional as F

from utils.model import calculate_positional_encoding, normalize_causal_params

class Categorical2d(Dataset):
    """Data generator for samples of p(a,b) from a categorical distribution
    with factorization p(a,b) = p(a)*p(b|a) or p(a,b) = p(b)*p(a|b)."""

    def __init__(self, factorization: str = "a->b", n_categories: int = 10, smoothness = 1, format: str  = "one-hot", beta: float = 1, rng: npr.Generator = None, seed: int =1234):
        """
        Initialize class
        Args:
            factorization (str): "a->b" for p(a,b) = p(a)*p(b|a); "b->a" for p(a,b) = p(b)*p(a|b); "a,b" for p(a,b) = p(a)*p(b). Effectively, "b->a" is implemented by swapping a and b before returning them.
            n_categories (int): The number of categories the categorical distributions p(a) and p(b) have. They will both have the same number of categories.
            smoothness (float): A factor controlling how smooth the distributions are. 0.1 is very spiky, 1 is intermediate, 5 is somewhat smooth, 20 is very smooth.
            format (str): "digit" to return regular digits e.g. [a, b] = [1, 5], or "one-hot" to return one-hot encoded values. "noise" to return a unique noise vector normed to 1 for each unique value of each variable.
            beta (float): A factor to multiply the denominator of the conditional with
            rng (numpy.random.Generator): A random number generator for generating data.
            seed (int): Seed to initialize a new rng if none is given.
        """
        self.factorization = factorization
        self.n_categories = n_categories
        self.smoothness = smoothness
        self.format = format

        # Initialize random number generator
        self.rng = rng if rng is not None else npr.default_rng(seed=seed)

        # Initialize distributions with Dirichlet priors
        self.p_A = self.rng.dirichlet(alpha = self.smoothness * np.ones(self.n_categories))
        self.p_BgivenA = self.rng.dirichlet(alpha = self.smoothness * np.ones(self.n_categories) / (beta*self.n_categories), size=self.n_categories)
        self.p_B = self.rng.dirichlet(alpha = self.smoothness * np.ones(self.n_categories)) # This is only used if a and b are independent

        if self.format == "noise":
            # If format is noise, initialize noise lookup vectors
            vecs = rng.normal(size=(2*self.n_categories, self.n_categories)) * 0.01
            base_vec_a = rng.normal(size=self.n_categories)
            self.a_vecs = base_vec_a + vecs[:self.n_categories]
            self.a_vecs = self.a_vecs / np.sum(self.a_vecs, axis=1)[:,None] # normalize
            base_vec_b = rng.normal(size=self.n_categories)
            self.b_vecs = base_vec_b + vecs[self.n_categories:]
            self.b_vecs = self.b_vecs / np.sum(self.b_vecs, axis=1)[:,None] # normalize

    def __len__(self) -> int:
        """
        Length of the dataset. It's dynamically generated, but to avoid infinite loops when
        working with this dataset, let's just set len() to a very large number
        """
        return 1e10
    
    def __getitem__(self, n) -> np.array:
        """
        Sample n samples from p(a,b) and return the data.
        (Note: Typically the argument here is idx and refers to the index of the data point, I've misused this functionality to indicate number of randomly sampled points)
        
        Note that if the factorization is "b->a", the code will proceed as if the factorization was "a->b" and at the end just swaps a and b, without loss of generality.

        Args:
            idx (int): ID of the data item in a set, gets ignored because we sample from distribution
        """
        a = self.rng.choice(self.n_categories, size=(n), p=self.p_A)

        if self.factorization == "a,b":
            # Sample b independently
            b = self.rng.choice(self.n_categories, size=(n), p=self.p_B)
        elif self.factorization in ("a->b", "b->a"):
            # Sample b given a, n times
            b = np.array([self.rng.choice(self.n_categories, p=self.p_BgivenA[a[i]]) for i in range(n)])
        else:
            raise ValueError('Factorization unknown !')

        # Change format of a and b if desired
        a, b = self.__format__(a, b, n)

        if self.factorization in ("a->b", "a,b"):
            return np.array([a,b]).transpose(1,0,2) # Use the transpose to get shape (n, 2, n_categories)
        if self.factorization == "b->a":
            # Swap a and b
            return np.array([b,a]).transpose(1,0,2)

    def __format__(self, a, b, n) -> tuple:
        """
        Format numbers a and b according to self.format
        Returns:
            (a, b): Set of a and b
            n: Number of data points
        """
        if self.format == "digit":
            return a, b
                
        elif self.format == "one-hot":
            a_onehot = np.eye(self.n_categories)[a]
            b_onehot = np.eye(self.n_categories)[b]
            return a_onehot, b_onehot
        
        elif self.format == "noise":
            return self.a_vecs[a], self.b_vecs[b]
        
        else:
            raise ValueError('Unknown data format requested!')

    def intervention(self) -> None:
        """
        Perform an intervention in the data generation process by randomly changing the distribution of one independent variable, p(a).
        Note that if factorization is "b->a", then this amounts to changing p(b) since __getitem__ eventually swaps a and b before returning them.
        """
        self.p_A = self.rng.dirichlet(alpha = self.smoothness * np.ones(self.n_categories))


class Causal2dModel(nn.Module):
    def __init__(self, model, n_categories: int, emb_size: int, temperature: float, use_bias: bool,
                 positional_encoding: str, positional_encoding_type: str, positional_encoding_scale: float,
                 hard: bool, normalization: str, use_input: bool =True,
                 initial_W_v: np.array= None):
        """
        Causal 2d model.
        Two variables (X, Y) are provided and the causal parameters X->X and Y->Y are thresholded with a Gumbel softmax 0 and 1.
        If X->X is 1, the factorization p(X,Y) = p(X)p(Y|X) is assumed to be correct and if Y->Y the factorization p(X,Y) = p(Y)p(X|Y) is assumed to be correct.
        Outputs are two variables X, Y (should be reconstructions) where the independent variable is sampled from a learned vector,
        and the dependent variable is sampled from a learned distribution p(dependent|independent) given the provided value of the independent variable.

        Args:
            n_categories: The inputs x to the model should have shape (batch_size, num_vars, n_categories), where n_categories is the dimensionality of one-hot encodings of the two variables contained in a datapoint x.
            emb_size: The embedding size that the vectors q and k will have. This argument can in the future be extended to have different dimensionalities for each.
            temperature: The tau value passed to gumble softmax, it controls the amount of noise used within the function.
            use_bias: Whether to add a bias to W_q, W_k and W_v. This argument can in the future be extended to have different bias booleans for each.
            positional_encoding: "None" if not used, "same" if both variables get the same encoding, "different" if they don't.
            positional_encoding_type: Whether to use a gaussian distribution or a linear ramp ('gauss' or 'ramp').
            positional_encoding_scale: What scale to use for pos encoding. Either std if 'gauss' or min/max if 'ramp'.
            hard: Whether to perform hard (True) or soft (False) softmax or Gumbel softmax, i.e. causal parameters are then either 0/1 or values between 0 and 1.
            normalization: Whether to use Gumbel, regular softmax or no normalization of causal parameters. Works together with parameter 'hard'.
            use_input: Whether to consider the model's input when calculating the output.


        Returns:
            outputs: Array containing logits for X and Y for all batch points
            causal_param_values: Raw causal parameters as they are after running the forward method; as numpy array of shape (2)
        """
        super().__init__()

        self.temperature = temperature
        self.hard = hard
        self.emb_size_v = emb_size
        self.use_input = use_input
        self.pos = calculate_positional_encoding(positional_encoding, positional_encoding_type, positional_encoding_scale, n_categories)
        self.normalization = normalization
        self.n_categories = n_categories
        self.model = model

        # Define query, key and value matrices as linear layers
        self.W_v = nn.Linear(n_categories, n_categories, bias=use_bias)

        self.causal_params = nn.Parameter(torch.Tensor([0.5, 0.5])) # shape (2) encodes causal parameters of first and second variable

        if model == "MM":
            self.independent_logits = torch.nn.Parameter(torch.ones(n_categories)/self.n_categories, requires_grad=True)


    def forward(self, x):
        # x shape: (batch_size, 2, n_categories)
        batch_size = x.size(0)

        # Expand positional encoding to batch size and add it to x
        pos_encoding = self.pos.repeat(batch_size, 1, 1)

        if self.use_input:
            encoded_x = x + pos_encoding
        else:
            encoded_x = pos_encoding
        encoded_x = encoded_x/torch.sum(encoded_x, axis=2)[:,:,None] # Normalize to 1
        
        # Reshape causal_params from shape (2) to shape (batch_size, 2, 1)
        causal_params_expanded = self.causal_params.repeat((batch_size, 1)).unsqueeze(2) # shape (batch_size, 2, 1)

        soft_causal_params = normalize_causal_params(causal_params_expanded, self.normalization, self.hard, self.temperature) # shape: (batch_size, 2, 1)

        # Flip to exchange W_v*x_v2 with W_v*x_1 because they should go into the reproduction of the respective other below.
        dependent_logits = torch.flip(self.W_v(encoded_x), dims=[1])

        if self.model == "MM":
            # Calculate outputs as x_1 = a_2*i and x_2 = a_1*i
            outputs = ((1-soft_causal_params) * self.independent_logits)
        elif self.model == "CM":
            # Calculate outputs as x_1 = a_2*W_v*x_2 and x_2 = a_1*W_v*x_1
            outputs = ((1-soft_causal_params) * dependent_logits)

        return outputs, causal_params_expanded.squeeze().detach().cpu().numpy(), soft_causal_params.squeeze().detach().cpu().numpy()


def run_experiment(config, df, setting_str, starttime_exp, path):
    # cfg is the config dict
    # df is the dataframe to which to append experiment data
    # setting_str is a string with the settings, used to match saved numpy arrays with the run
    # starttime_exp is a string of the start time of the experiment, to be saved in the dataframe
    # path of folder where experiments are to be saved

    cfg = SimpleNamespace(**config)

    # Initialize random number generator
    rng = npr.default_rng(seed=None)

    loss_fn = torch.nn.CrossEntropyLoss(reduction='none')

    ############## Initialize logging arrays ##############
    probability_matrices = np.zeros((cfg.num_runs, cfg.num_epochs, cfg.num_batches, 4, cfg.n_categories, cfg.n_categories))
    probability_vectors = np.zeros((cfg.num_runs, cfg.num_epochs, cfg.num_batches, 2, cfg.n_categories))
    raw_causal_params = np.zeros((cfg.num_runs, cfg.num_epochs, cfg.num_batches, 2))
    soft_causal_params = np.zeros((cfg.num_runs, cfg.num_epochs, cfg.num_batches, 2))
    train_loss = np.zeros((cfg.num_runs, cfg.num_epochs, cfg.num_batches))
    eval_loss = np.zeros((cfg.num_runs, cfg.num_epochs, cfg.num_batches, 2)) # Stores separate loss for A and B reconstruction
    causal_param_gradients = np.zeros((cfg.num_runs, cfg.num_epochs, cfg.num_batches, 2))
    ########################################################

    # Training loop
    for run_id in tqdm(range(cfg.num_runs)):

        # Initialize new dataset for training
        data_generator = Categorical2d(factorization=cfg.factorization, smoothness=cfg.smoothness,
                                       n_categories=cfg.n_categories, format=cfg.format, beta=cfg.beta, rng=rng)

        # Initialize new model
        model = Causal2dModel(model=cfg.model, n_categories=cfg.n_categories, emb_size=cfg.emb_size,
                                            temperature=cfg.temperature, use_bias=cfg.use_bias, hard=cfg.hard,
                                            positional_encoding=cfg.positional_encoding,
                                            positional_encoding_type=cfg.pos_enc_type,
                                            positional_encoding_scale=cfg.pos_enc_scale,
                                            normalization=cfg.normalization, use_input=cfg.use_input)
        

        # Initialize optimizer
        optimizer = torch.optim.Adam(model.parameters(), lr=cfg.lr)

        # Generate a list of batches from the data_generator
        batched_data = [torch.Tensor(data_generator[cfg.batch_size]) for _ in range(cfg.num_batches)]

        # Train for a specified number of epochs
        for i in range(cfg.num_epochs):

            # Train on all batches consecutively
            for j in range(cfg.num_batches):

                p_B = data_generator.p_A @ data_generator.p_BgivenA # This is different to the also existing variable data_generator.p_B, which is however not used for generating data since it's not part of the factorization.
                p_AgivenB = 1/p_B[:,None] * data_generator.p_A * data_generator.p_BgivenA.T

                probability_matrices[run_id, i, j, 0] = data_generator.p_BgivenA                                         # p_BgivenA
                probability_matrices[run_id, i, j, 1] = p_AgivenB                                                        # p_AgivenB
                probability_matrices[run_id, i, j, 2] = model.W_v.weight.detach().cpu().numpy().T
                if cfg.model == "joint_Wi":
                    probability_matrices[run_id, i, j, 3] = model.W_i.weight.detach().cpu().numpy().T
                probability_vectors[run_id, i, j, 0] = data_generator.p_A
                probability_vectors[run_id, i, j, 1] = p_B

                ########## Train model ##########
                # Get a batch of datapoints
                inputs = batched_data[j]

                optimizer.zero_grad()

                # Pass inputs through model
                logits, train_causal_params, train_soft_causal_params = model(inputs)

                if cfg.loss == "marginal":
                    # Transpose logits and inputs in order to meet input dimensionality of CELoss
                    logits = logits.transpose(1, 2)
                    targets = inputs.transpose(1, 2)
                
                elif cfg.loss == "joint":
                    kronecker_product = lambda x1, x2: torch.einsum("nk,nl->nkl",x1,x2).reshape(cfg.batch_size, -1) #can't just use torch.kron here since it doesn't allow specifying dimensions
                    logits = kronecker_product(logits[:,0,:], logits[:,1,:])
                    targets = kronecker_product(inputs[:,0,:], inputs[:,1,:])

                # Calculate loss, backwards pass and parameter updates
                loss = torch.mean(loss_fn(logits, targets))
                loss.backward()
                optimizer.step()

                ############## Log general data ##############
                train_loss[run_id, i, j] = np.mean(loss.detach().cpu().numpy()) # Average train losses over all batches for this epoch and store for plotting
                eval_loss[run_id, i, j], eval_causal_params, eval_soft_causal_params = evaluate(model, data_generator, loss_fn, cfg.loss, cfg.eval_size)

                causal_param_gradients[run_id, i, j] = (model.causal_params.grad).detach().cpu().numpy()

                raw_causal_params[run_id, i, j] = np.mean(train_causal_params, axis=0)
                soft_causal_params[run_id, i, j] = np.mean(softmax(train_causal_params, axis=1), axis=0)

            # Potentially intervene and sample new data
            if (i+1) % cfg.distribution_change_freq == 0:
                data_generator.intervention()
                batched_data = [torch.Tensor(data_generator[cfg.batch_size]) for _ in range(cfg.num_batches)]

    exp_results = {"pAs": probability_vectors[:,:,:,0],
                    "pBs": probability_vectors[:,:,:,1],
                    "pBgivenAs": probability_matrices[:,:,:,0],
                    "pAgivenBs": probability_matrices[:,:,:,1],
                    "Wvs": probability_matrices[:,:,:,2],
                    "causal_params": raw_causal_params,
                    "soft_causal_params": soft_causal_params,
                    "train_loss": train_loss,
                    "eval_loss": eval_loss}
    if cfg.model == "joint_Wi":
        # Also store Wis
        exp_results["Wis"] = probability_matrices[:,:,:,3]
    for key, value in exp_results.items():
        np.save(f"{path}/{setting_str}_{key}.npy", value)
    df = pd.concat([df, pd.DataFrame([{"name": setting_str, "time": starttime_exp} | config])], ignore_index=True)
    return df

params = dict(
    model = "placeholder", # "MM", "CM"
    temperature = 2, # temperature of the gumbel softmax
    normalization = "gumbel", # Which normalization to apply. Choices "softmax", "gumbel", "none"
    hard = False, # Whether to use hard thresholding in gumbel/softmax
    num_runs = 100,
    num_epochs = 300,
    num_batches = 5,
    batch_size = 128,
    eval_size = 5, # On how many datapoints to evaluate the model after each epoch
    distribution_change_freq = 1, # Number of epochs before the prior p(A) is changed. If this is > num_epochs there won't be a change in the distribution
    n_categories = 5, # Categories of each one-hot encoded data point
    use_bias=False, # Whether to add a bias term to Q, K and V
    emb_size=10, # The embedding size of matrices Q, K, V
    positional_encoding = "placeholder", # 'none', 'same', 'different'
    pos_enc_type = "uniform", # 'gauss' or 'ramp', 'uniform' for positional_encoding="same"
    pos_enc_scale=1, # std for pos_enc_type='gauss' and min/max for pos_enc_type='ramp'
    lr=0.1, # learning rate,
    use_input=False, # Whether to provide A,B input to the model,
    record_video=False, # Whether to record a video visualization of the matrices
    factorization="a->b", #"a->b", "b->a", "a,b"
    smoothness=1,
    format="one-hot",#"one-hot", "noise", "digit"
    loss="placeholder",#"marginal", "joint"
    beta=1,
)

##################################################################################
# Actual program:

model = "CM" # "MM", "CM"
data = "interventional" # "observational", "interventional"
use_input = "input" # "input", "noinput"

# Observational or interventional?
if data == "observational":
    params["distribution_change_freq"] = 1000
    params["epochs"] = 300
elif data == "interventional":
    params["distribution_change_freq"] = 1
    params["epochs"] = 300

# Model
params["model"] = model
if model == "CM":
    params["loss"] = "marginal"
    params["batch_size"] = 128
    params["hard"] = False
    params["num_batches"] = 5
    params["temperature"] = 2
elif model == "MM":
    params["loss"] = "marginal"
    params["batch_size"] = 128
    params["hard"] = False
    params["num_batches"] = 5
    params["temperature"] = 2

# Input
if use_input=="input":
    params["positional_encoding"] = "none"
    params["use_input"] = True
elif use_input == "noinput":
    params["positional_encoding"] = "same"
    params["use_input"] = False


starttime_all = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
path = f"data/{starttime_all}_{model}_{use_input}_{data}"
Path(path).mkdir(parents=True, exist_ok=True)
df = pd.DataFrame()

factors = np.geomspace(0.1,10,21)
for i, f in enumerate(factors):
    starttime_exp = datetime.now().strftime("%Y-%m-%d-%H-%M-%S") # String used for creating a folder with plots.
    print(f"Factor: {f:.2f}")
    params["beta"]=f
    setting_str = f"{model}_{use_input}_{data}_{i}" # make sure the setting string contains i as an identifier to be able to match arrays to index in dataframe
    df = run_experiment(params, df, setting_str, starttime_exp, path)
df.to_csv(f"{path}/df_{model}_{use_input}_{data}")
