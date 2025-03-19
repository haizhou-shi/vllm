import torch
import math
from torch.nn import functional as F
from dataclasses import dataclass

@dataclass
class SparseLogits:
    sparse_logits: torch.Tensor # shape: [n_samples, [seq_len, k]]
    k: int # number of top logits to keep
    vocab_size: int

    @property
    def shape(self):
        return self.sparse_logits.shape
    
    @classmethod
    def from_dense_tensor(cls, logits, k=200):
        """
        Convert dense logits tensor to sparse format containing only top-k values.
        
        Args:
            logits: Dense logits tensor of shape [n_samples, seq_len, vocab_size]
            k: Number of top logits to keep
        
        Returns:
            A sparse representation containing the top-k values and their indices
        """
        # Get the dimensions
        device = logits.device
        n_samples, seq_len, vocab_size = logits.shape
        
        # Get the top-k logits and their indices
        # Here we add another slot (k+1) for storing the remaining logits value.
        # We don't care about the actual indices 
        logits_values, logits_indices = torch.topk(logits, k=k, dim=-1)  # shape: [n_samples, seq_len, k]

        # Average Logits Remaining. 
        # NOTE: here we cannot average the probs after 
        #   the softmax for the sake of memory efficiency.
        #   this will cause the under-estimation of the uncertainty,
        #   since exp() is convex.
        # NOTE: the larger the remaining mass of the logits, the more under-estimation
        #   of the uncertainty.
        # TODO: better way to estimate.
        logits_remain_mean = (logits.sum(dim=-1) - logits_values.sum(dim=-1)) / (vocab_size - k)

        # append the mean of the remaining logits to the last index.
        # the filld value is the log(vocab_size - k) + logits_remain_mean, 
        # which is the log probability of the remaining logits.
        # now the shape of logits_values is [n_samples, seq_len, vocab_size+1]
        # and it can be directly used for uncertainty estimation.
        # logits_values = torch.cat([logits_values, -100 * torch.ones_like(logits_remain_mean.unsqueeze(-1))], dim=-1)
        logits_values = torch.cat([logits_values, math.log(vocab_size - k) + logits_remain_mean.unsqueeze(-1)], dim=-1)
        logits_indices = torch.cat([logits_indices, torch.full_like(logits_indices[:, :, :1], vocab_size, dtype=logits_indices.dtype)], dim=-1)
        
        # Create batch indices and sequence position indices
        batch_indices = torch.arange(n_samples).view(-1, 1, 1).expand(-1, seq_len, k+1).to(device)
        seq_indices = torch.arange(seq_len).view(1, -1, 1).expand(n_samples, -1, k+1).to(device)
        
        # Create indices for sparse tensor
        indices = torch.stack([
            batch_indices.reshape(-1),
            seq_indices.reshape(-1),
            logits_indices.reshape(-1)
        ], dim=0)
        
        # Create sparse tensor
        sparse_logits = torch.sparse_coo_tensor(
            indices=indices,
            # float() is necessary here as sparse.softmax is not imlemented for bf16.
            values=logits_values.reshape(-1).float(), 
            size=(n_samples, seq_len, vocab_size+1),
        ).coalesce()
        
        return cls(
            sparse_logits=sparse_logits, 
            # logits_remain_mean=logits_remain_mean, 
            k=k,
            vocab_size=vocab_size,
        )
    
    def softmax(self, mean=False):
        """
        Compute the softmax of the sparse logits. 

        Args:
            mean: If True, compute the mean of the soft_max of the logits. Otherwise, compute the softmax of the logits.
        """
        softmaxed = torch.sparse.softmax(self.sparse_logits, dim=-1)
        if not mean:
            return softmaxed
        return torch.sparse.sum(softmaxed, dim=0) / self.shape[0]

    def log_softmax(self, mean=False):
        """
        Compute the softmax of the sparse logits. 

        Args:
            mean: If True, compute the mean of the soft_max of the logits. Otherwise, compute the softmax of the logits.
        """
        
        if mean: # log(softmax().mean())
            softmaxed_mean = self.softmax(mean=True)
            # This is a workaround for the lack of log() function for sparse tensors,
            # as log() is non zero-preserving (log(0) = -inf ≠ 0): https://discuss.pytorch.org/t/logarithm-of-a-sparse-tensor/206958
            # here we use log1p(x-1)=log(x) to compute the log of the softmaxed_mean.
            ones_like = torch.sparse_coo_tensor(
                indices=softmaxed_mean.indices(),
                values=torch.ones_like(softmaxed_mean.values()),
                size=softmaxed_mean.size(),
            ).coalesce()
            res = torch.log1p(softmaxed_mean - ones_like)
        else:
            res = torch.sparse.log_softmax(self.sparse_logits, dim=-1)
        
        # there is some risk that log(~0) = -inf, so we replace it with 0 
        # to make it contribute to nothing to the final uncertainty, as
        # 0 * log(0) = 0.
        res.values()[res.values().isinf()] = 0.

        return res
    
    def total_uncertainty(self):
        """
        Compute the total uncertainty of the sparse logits.
        """
        probs = self.softmax(mean=True)
        log_probs = self.log_softmax(mean=True)
        return self.entropy(probs, log_probs)

    def aleatoric_uncertainty(self):
        """
        Compute the aleatoric uncertainty of the sparse logits.
        """
        probs = self.softmax(mean=False)
        log_probs = self.log_softmax(mean=False)
        entropy = torch.cat([self.entropy(prob, log_prob).unsqueeze(0) for prob, log_prob in zip(probs, log_probs)], dim=0).mean(dim=0)
        return entropy

    def epistemic_uncertainty(self):
        """
        Compute the epistemic uncertainty of the sparse logits.
        """
        return self.total_uncertainty() - self.aleatoric_uncertainty()

    def entropy(self, probs, log_probs):
        """
        Compute the entropy of the sparse logits.
        """
        return (-torch.sparse.sum(probs * log_probs, dim=-1)).to_dense()

    def evaluate_uncertainty_all(self):
        """
        Evaluate all types of uncertainty.
        """
        tu = self.total_uncertainty()
        au = self.aleatoric_uncertainty()
        return tu, au, tu - au

def entropy(logits):
    """
    Entropy calculation for the logits.
    """
    probs = F.softmax(logits, dim=-1)
    log_probs = F.log_softmax(logits, dim=-1)
    return -torch.sum(probs * log_probs, dim=-1)

def total_uncertainty(logits):
    """
    H(E_{\theta}[P(y|x, \theta)]), total uncertainty evaluation for the logits.
    """
    mean_logits = torch.log(F.softmax(logits, dim=-1).mean(dim=0) + 1e-8) # shape: [seq_len, vocab_size]
    token_entropy = entropy(mean_logits) # shape: [seq_len]
    return token_entropy # shape: [seq_len]

def aleatoric_uncertainty(logits):
    """
    E_{\theta}[H(P(y|x, \theta))], aleotoric uncertainty evaluation for the logits.
    """
    token_entropy = entropy(logits) # shape: [n_samples, seq_len]
    return token_entropy.mean(dim=0) # shape: [seq_len]
    
def evaluate_uncertainty_all(logits):
    """
    Uncertainty evaluation for the logits.
        TU: total uncertainty, H(E_{\theta}[P(y|x, \theta)])
        AU: aleotoric uncertainty, E_{\theta}[H(P(y|x, \theta))]
        EU: epistemic uncertainty, TU - AU
    Input: 
        logits: [n_samples, seq_len, vocab_size]
        labels: [seq_len]
        uncertainty_type: the type of uncertainty to evaluate.
    """
    
    tu = total_uncertainty(logits)
    au = aleatoric_uncertainty(logits)
    eu = tu - au
    return tu.cpu().tolist(), au.cpu().tolist(), eu.cpu().tolist()