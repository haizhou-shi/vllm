import torch
from torch.nn import functional as F

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

def aleotoric_uncertainty(logits):
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
    au = aleotoric_uncertainty(logits)
    eu = tu - au
    return tu.cpu().tolist(), au.cpu().tolist(), eu.cpu().tolist()