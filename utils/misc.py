# File for miscellaneous functions

import numpy as np
import torch
from torch.utils.data import Dataset
from torch import nn
from pathlib import Path
import pprint
import git
import sys


def evaluate(model: nn.Module, data_generator: Dataset, loss_fn, loss, eval_size: int):
    """
    Function to evaluate a model with newly sampled data from the data_generator.

    Arguments:
        model: The model object.
        data_generator: A torch dataset object.
        loss_fn: The torch loss function class used for the model
        loss: What type of loss to use, either "marginal" or "joint"
        eval_size: The number of sampled points to use for evaluation (similar to batch_size during training).
    Return:
        Loss (float): The loss averaged over eval_size data points
        causal_params (np.ndarray): Array of shape (2) with causal parameters returned by the model
    """
    inputs = torch.Tensor(data_generator[eval_size])
    model.eval()
    logits, causal_params, soft_causal_params = model(inputs)

    if loss == "marginal":
        # Transpose logits and inputs in order to meet input dimensionality of CELoss
        logits = logits.transpose(1, 2)
        targets = inputs.transpose(1, 2)
    
    elif loss == "joint":
        kronecker_product = lambda x1, x2: torch.einsum("nk,nl->nkl",x1,x2).reshape(x1.shape[0], -1) #can't just use torch.kron here since it doesn't allow specifying dimensions
        logits = kronecker_product(logits[:,0,:], logits[:,1,:])
        targets = kronecker_product(inputs[:,0,:], inputs[:,1,:])
    
    # Calculate loss
    loss = torch.mean(loss_fn(logits, targets), axis=0)

    model.train()

    return loss.detach().cpu().numpy(), causal_params, soft_causal_params

def write_parameters(starttime, params):
    """
    A small function to write training parameters into a text file for a given experiment (i.e. collection of runs).
    
    This currently only saves a part of the parameters, see the list of arguments.
    Also it would probably be more elegant to have parameters as a dict and save the dict in json format. This is a ToDo for later.

    Args:
        starttime: String with time of the current run
        params: The dictionary of parameters read from the config file
    """
    Path(f"plots/{starttime}").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    repo = git.Repo(search_parent_directories=True)
    sha = repo.head.object.hexsha
    branch = repo.active_branch.name.split("/")[-1]
    with open(f"plots/{starttime}/parameters.txt", "w") as text_file:
        text_file.write(f"Parameters used in this experiment (timestamp {starttime}):\n\n")
        text_file.write(f"Current branch: {branch}\n")
        text_file.write(f"Git hash: {sha}\n\n")
        pprint.pprint(params, text_file)
        text_file.write("\n\n\n\n\n\n\n\n")
        text_file.write(f"Diff between commit stated above and code that is currently executed:\n\n{repo.git.diff()}")

def save_data(starttime, all_soft_causal_params, all_causal_params):
    """
    Saves the numpy arrays with causal parameters that resulted from an experiment (i.e. all runs)

    Args:
        starttime: String with time of the current run
    """
    Path(f"plots/{starttime}").mkdir(parents=True, exist_ok=True) # Create folder if it doesn't exist
    np.save(f"plots/{starttime}/all_soft_causal_params.npy", all_soft_causal_params)
    np.save(f"plots/{starttime}/all_causal_params.npy", all_causal_params)

def print_progress(i, n):
    """
    A function to create a printout about the progress in plotting.create_visualization_video()
    """
    print(f'Saving frame {i} of {n}', end='\r')
    sys.stdout.flush()
