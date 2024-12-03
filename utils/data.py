# File for dataset classes

from torch.utils.data import Dataset
import numpy as np
import numpy.random as npr

class Categorical2d(Dataset):
    """Data generator for samples of p(a,b) from a categorical distribution
    with factorization p(a,b) = p(a)*p(b|a) or p(a,b) = p(b)*p(a|b)."""

    def __init__(self, factorization: str = "a->b", n_categories: int = 10, smoothness = 1, format: str  = "one-hot", epsilon: float = 1, rng: npr.Generator = None, seed: int =1234):
        """
        Initialize class
        Args:
            factorization (str): "a->b" for p(a,b) = p(a)*p(b|a); "b->a" for p(a,b) = p(b)*p(a|b); "a,b" for p(a,b) = p(a)*p(b). Effectively, "b->a" is implemented by swapping a and b before returning them.
            n_categories (int): The number of categories the categorical distributions p(a) and p(b) have. They will both have the same number of categories.
            smoothness (float): A factor controlling how smooth the distributions are. 0.1 is very spiky, 1 is intermediate, 5 is somewhat smooth, 20 is very smooth.
            format (str): "digit" to return regular digits e.g. [a, b] = [1, 5], or "one-hot" to return one-hot encoded values. "noise" to return a unique noise vector normed to 1 for each unique value of each variable.
            epsilon (float): A factor to multiply the denominator of the conditional with
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
        self.p_BgivenA = self.rng.dirichlet(alpha = self.smoothness * np.ones(self.n_categories) / (epsilon*self.n_categories), size=self.n_categories)
        # self.p_BgivenA = np.array([[0.05, 0.05, 0.8, 0.05, 0.05],
        #                          [0.05, 0.75, 0.05, 0.1, 0.05],
        #                          [0.05, 0.75, 0.05, 0.1, 0.05],
        #                          [0.05, 0.05, 0.05, 0.05, 0.8],
        #                          [0.8, 0.05, 0.05, 0.05, 0.05]])
        #self.p_BgivenA = skewed_matrix(self.n_categories, 0.1)
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
        # self.p_B = self.rng.dirichlet(alpha = self.smoothness * np.ones(self.n_categories))
        # self.p_BgivenA = self.rng.dirichlet(alpha = self.smoothness * np.ones(self.n_categories)  / self.n_categories, size=self.n_categories)