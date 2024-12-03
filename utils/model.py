# File with neural network models
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np


def calculate_positional_encoding(pos_enc, pos_enc_type, pos_enc_scale, n_categories):
    if pos_enc == 'same':
        if pos_enc_type == 'gauss':
            posenc_1D = torch.randn(n_categories) * pos_enc_scale
        elif pos_enc_type == 'ramp':
            posenc_1D = torch.linspace(-pos_enc_scale, pos_enc_scale, n_categories)
        elif pos_enc_type == 'uniform':
            posenc_1D = torch.ones(n_categories) * 0.1
        elif pos_enc_type == 'one-hot':
            posenc_1D = torch.eye(n_categories)[4]
        pos = posenc_1D.repeat(2, 1)
    elif pos_enc == 'different':
        if pos_enc_type == 'gauss':
            pos = torch.randn((2, n_categories)) * pos_enc_scale
        elif pos_enc_type == 'ramp':
            ramp = torch.linspace(-pos_enc_scale, pos_enc_scale, n_categories)[None, :]
            pos = torch.cat([ramp, -ramp], axis=0)
        elif pos_enc_type == 'one-hot':
            pos = torch.cat([torch.eye(n_categories)[4][None, :], torch.eye(n_categories)[7][None, :]], axis=0)
    elif pos_enc == 'none':
        pos = torch.zeros(2, n_categories)
    pos.requires_grad = False  # Turn off gradients again manually just to make sure.
    return pos

def normalize_causal_params(causal_params, normalization, hard, temperature):
    # soft_causal_params must have shape (batch_size, 2, 1)
    if normalization == "none":
        soft_causal_params = causal_params
    elif normalization == "gumbel":
        soft_causal_params = F.gumbel_softmax(causal_params, tau=temperature, dim=1, hard=hard)
    elif normalization == "softmax":
        # Scale by temperature and add tiny noise to avoid always selecting first if both values are equal:
        causal_params = causal_params/temperature + torch.rand_like(causal_params) * 1e-6
        soft_causal_params = F.softmax(causal_params, dim=1)
        if hard:
            hard_scores = torch.zeros_like(soft_causal_params).scatter_(1, soft_causal_params.max(1, keepdim=True)[1], 1.0)
            soft_causal_params = hard_scores - soft_causal_params.detach() + soft_causal_params
    else:
        raise NotImplementedError(f"Normalization '{normalization}' is not implemented!")
    return soft_causal_params # shape (batch_size, 2, 1)


class Causal2dModel(nn.Module):
    def __init__(self, model: str, n_categories: int, emb_size: int, temperature: float, use_bias: bool, positional_encoding: str, positional_encoding_type: str,
                 positional_encoding_scale: float, hard: bool, normalization: str, use_input: bool =True):
        """
        Causal 2d model.
        Two variables (X, Y) are provided and the causal parameters X->X and Y->Y are thresholded with a Gumbel softmax 0 and 1.
        If X->X is 1, the factorization p(X,Y) = p(X)p(Y|X) is assumed to be correct and if Y->Y the factorization p(X,Y) = p(Y)p(X|Y) is assumed to be correct.
        Outputs are two variables X, Y (should be reconstructions) where the independent variable is sampled from a learned vector,
        and the dependent variable is sampled from a learned distribution p(dependent|independent) given the provided value of the independent variable.

        Args:
            model: Name of the model
            n_categories: The inputs x to the model should have shape (batch_size, num_vars, n_categories), where n_categories is the dimensionality of one-hot encodings of the two variables contained in a datapoint x.
            emb_size: The embedding size that the vectors q and k will have. This argument can in the future be extended to have different dimensionalities for each.
            temperature: The tau value passed to gumble softmax, it controls the amount of noise used within the function.
            use_bias: Whether to add a bias to W_q, W_k and W_v. This argument can in the future be extended to have different bias booleans for each.
            positional_encoding: "None" if not used, "same" if both variables get the same encoding, "different" if they don't.
            positional_encoding_type: Whether to use a gaussian distribution or a linear ramp ('gauss' or 'ramp').
            positional_encoding_scale: What scale to use for pos encoding. Either std if 'gauss' or min/max if 'ramp'.
            hard: Whether to perform hard (True) or soft (False) softmax or Gumbel softmax, i.e. causal parameters are then either 0/1 or values between 0 and 1.
            normalization: Whether to use Gumbel, regular softmax or no normalization of soft causal parameter values. Works together with parameter 'hard'.
            use_input: Whether to consider the model's input when calculating the output.


        Returns:
            outputs: Array containing logits for X and Y for all batch points
            causal_param_values: Raw causal parameter values as they are after running the forward method; as numpy array of shape (2)
        """
        super().__init__()

        self.model = model
        self.temperature = temperature
        self.hard = hard
        self.emb_size_v = emb_size
        self.use_input = use_input
        self.pos = calculate_positional_encoding(positional_encoding, positional_encoding_type, positional_encoding_scale, n_categories)
        self.normalization = normalization
        self.n_categories = n_categories

        # Define query, key and value matrices as linear layers
        self.W_v = nn.Linear(n_categories, n_categories, bias=use_bias)

        if initial_W_v is not None:
            initial_W_v = torch.from_numpy(initial_W_v).float()
            logits = torch.log(initial_W_v.T)

            self.W_v.weight = nn.Parameter(logits)

        self.causal_params = nn.Parameter(torch.Tensor([0.5, 0.5])) # shape (2) encodes causal parameters for first and second variable

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
