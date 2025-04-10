import torch
import math
from torch.nn import functional as F
from dataclasses import dataclass

def softmax_inplace(logits):
    """
    Inplace softmax for the logits.
    """
    logits.sub_(torch.max(logits, dim=-1, keepdim=True).values)
    logits.exp_().div_(logits.sum(dim=-1, keepdim=True))
    
    return logits

@dataclass
class SparseProbs:
    sparse_probs: torch.Tensor # shape: [n_samples, [seq_len, k]]
    remaining_vocab_sizes: torch.Tensor # [n_samples, seq_len]
    remaining_probs: torch.Tensor # [n_samples, seq_len] 
    vocab_size: int

    @property
    def shape(self):
        return self.sparse_probs.shape
    
    @classmethod
    def from_dense_logits_top_p(cls, logits, p=0.9):
        """
        Convert dense logits tensor to sparse format containing only top-k values.
        
        Args:
            logits: Dense logits tensor of shape [n_samples, seq_len, vocab_size]
            k: Number of top logits to keep
        
        Returns:
            A sparse representation containing the top-k values and their indices
        """

        logits = softmax_inplace(logits)
        
        # Get the dimensions
        device = logits.device
        n_samples, seq_len, vocab_size = logits.shape
        
        # Get the top-p logits and their indices
        # Here we add another slot (k+1) for storing the remaining logits value.
        # We don't care about the actual indices 
        # TODO:
        #   The big for loop + while loop is not time-efficient.
        #   How to further optimize this? 
        logits_values, logits_indices, remaining_vocab_sizes, remaining_probs = [], [], [], []
        for i in range(seq_len):
            # for one token, we use a shared k.
            k = 1000
            sampled_logits_i = logits[:, i, :]
            sum_logits_i = torch.sum(sampled_logits_i)
            while True:
                if k > vocab_size:
                    k = vocab_size
                logits_values_i, logits_indices_i = torch.topk(sampled_logits_i, k=k, dim=-1)  # shape: [n_samples, seq_len, k]
                if torch.sum(logits_values_i) / sum_logits_i >= p:
                    # Average Logits Remaining. 
                    # NOTE: 
                    #   here we directly sum the remaining probs and put it to the last index.
                    #   this will cause the under-estimation of the uncertainty.
                    #   the larger the remaining mass of the logits, the more under-estimation
                    #   of the uncertainty.
                    # NOTE: 
                    #   we compensate for the under-estimation by adding the cross entropy p * log(K)
                    #   to the final uncertainty, where K is the number of remaining logits.
                    #   this is equivalent to assigning the mean of the probs to the rest of the indices.
                    logits_remain_mean_i = (sampled_logits_i.sum(dim=-1, keepdims=True) - logits_values_i.sum(dim=-1, keepdims=True))
                    remaining_vocab_sizes.append(torch.full_like(logits_remain_mean_i, vocab_size - k))
                    remaining_probs.append(logits_remain_mean_i)

                    # append the mean of the remaining logits to the last index.
                    # the filld value is the log(vocab_size - k) + logits_remain_mean, 
                    # which is the log probability of the remaining logits.
                    # now the shape of logits_values is [n_samples, seq_len, vocab_size+1]
                    # and it can be directly used for uncertainty estimation.
                    logits_values_i = torch.cat([logits_values_i, logits_remain_mean_i], dim=-1)
                    logits_indices_i = torch.cat([logits_indices_i, torch.full_like(logits_indices_i[:, :1], vocab_size, dtype=logits_indices_i.dtype)], dim=-1)
                    logits_values.append(logits_values_i)
                    logits_indices.append(logits_indices_i)
                    break
                k += 1000
        
        # Since k might be different for each position, we need to handle this differently
        # Instead of stacking, we'll create COO indices directly
        
        # Create lists to hold all indices and values
        all_batch_indices = []
        all_seq_indices = []
        all_vocab_indices = []
        all_values = []
        
        # For each position in the sequence
        for seq_idx, (values_i, indices_i) in enumerate(zip(logits_values, logits_indices)):
            # For each sample in the batch
            for batch_idx in range(n_samples):
                # Get number of values for this specific (batch, seq) pair
                num_values = values_i[batch_idx].size(0)
                
                # Add batch indices
                all_batch_indices.append(torch.full((num_values,), batch_idx, device=device))
                
                # Add sequence indices
                all_seq_indices.append(torch.full((num_values,), seq_idx, device=device))
                
                # Add vocabulary indices
                all_vocab_indices.append(indices_i[batch_idx])
                
                # Add values
                all_values.append(values_i[batch_idx])
        
        # Concatenate all indices and values
        batch_indices = torch.cat(all_batch_indices)
        seq_indices = torch.cat(all_seq_indices)
        vocab_indices = torch.cat(all_vocab_indices)
        values = torch.cat(all_values)
        
        # Create the indices tensor for the sparse COO tensor
        indices = torch.stack([
            batch_indices,  # batch dimension
            seq_indices,    # sequence dimension
            vocab_indices   # vocabulary dimension
        ], dim=0)

        # Create the remaining vocab sizes tensor
        # for uncertainty estimation compensation.
        remaining_vocab_sizes = torch.cat(remaining_vocab_sizes, dim=1)
        remaining_probs = torch.cat(remaining_probs, dim=1)
        
        # Create sparse tensor
        sparse_logits = torch.sparse_coo_tensor(
            indices=indices,
            # float() is necessary here as sparse.softmax is not implemented for bf16.
            values=values.float(),
            size=(n_samples, seq_len, vocab_size+1),
        ).coalesce()
        
        return cls(
            sparse_probs=sparse_logits, 
            # logits_remain_mean=logits_remain_mean, 
            remaining_vocab_sizes=remaining_vocab_sizes,
            remaining_probs=remaining_probs,
            vocab_size=vocab_size,
        )
    
    def softmax(self, mean=False):
        """
        Compute the softmax of the sparse logits. 

        Args:
            mean: If True, compute the mean of the soft_max of the logits. Otherwise, compute the softmax of the logits.
        """
        if not mean:
            return self.sparse_probs
        return torch.sparse.sum(self.sparse_probs, dim=0) / self.shape[0]

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
            probs = self.sparse_probs
            ones_like = torch.sparse_coo_tensor(
                indices=probs.indices(),
                values=torch.ones_like(probs.values()),
                size=probs.size(),
            ).coalesce()
            res = torch.log1p(probs - ones_like)
        
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
        entropy = self.entropy(probs, log_probs)

        # p * log(K) compensation for the under-estimation of the uncertainty.
        # NOTE: this is still under-estimation, but it's better than nothing. 
        #    the two compensates are the same, as we have the same vocab size
        #    for each sample.
        compensate = torch.log(self.remaining_vocab_sizes.mean(dim=0)) * self.remaining_probs.mean(dim=0)
        return entropy + compensate

    def aleatoric_uncertainty(self):
        """
        Compute the aleatoric uncertainty of the sparse logits.
        """
        probs = self.softmax(mean=False)
        log_probs = self.log_softmax(mean=False)
        entropy = torch.cat([self.entropy(prob, log_prob).unsqueeze(0) for prob, log_prob in zip(probs, log_probs)], dim=0).mean(dim=0)

        # p * log(K) compensation for the under-estimation of the uncertainty.
        # NOTE: this is still under-estimation, but it's better than nothing.
        #    the two compensates are the same, as we have the same vocab size
        #    for each sample.
        compensate = (torch.log(self.remaining_vocab_sizes) * self.remaining_probs).mean(dim=0)
        return entropy + compensate

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
        return tu.cpu().tolist(), au.cpu().tolist(), (tu - au).cpu().tolist()


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
    
    @classmethod
    def from_dense_tensor_top_p(cls, logits, p=0.4):
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

        # top-p means:
        #   sum(top_logits) / sum(all_logits) >= p,
        # hence we need to set the minimum value to 0,
        # the remaining logits are all > 0. 
        logits -= torch.min(logits, dim=-1, keepdim=True).values
        
        # Get the top-k logits and their indices
        # Here we add another slot (k+1) for storing the remaining logits value.
        # We don't care about the actual indices 
        logits_values, logits_indices = [], []
        for i in range(seq_len):
            k = 1000
            sampled_logits_i = logits[:, i, :]
            sum_logits_i = torch.sum(sampled_logits_i)
            while True:
                if k > vocab_size:
                    k = vocab_size
                logits_values_i, logits_indices_i = torch.topk(sampled_logits_i, k=k, dim=-1)  # shape: [n_samples, seq_len, k]
                if torch.sum(logits_values_i) / sum_logits_i >= p:
                    # Average Logits Remaining. 
                    # NOTE: here we cannot average the probs after 
                    #   the softmax for the sake of memory efficiency.
                    #   this will cause the under-estimation of the uncertainty,
                    #   since exp() is convex.
                    # NOTE: the larger the remaining mass of the logits, the more under-estimation
                    #   of the uncertainty.
                    # TODO: better way to estimate.
                    logits_remain_mean_i = (sampled_logits_i.sum(dim=-1) - logits_values_i.sum(dim=-1)) / (vocab_size - k)

                    # append the mean of the remaining logits to the last index.
                    # the filld value is the log(vocab_size - k) + logits_remain_mean, 
                    # which is the log probability of the remaining logits.
                    # now the shape of logits_values is [n_samples, seq_len, vocab_size+1]
                    # and it can be directly used for uncertainty estimation.
                    logits_values_i = torch.cat([logits_values_i, math.log(vocab_size - k) + logits_remain_mean_i.unsqueeze(-1)], dim=-1)
                    logits_indices_i = torch.cat([logits_indices_i, torch.full_like(logits_indices_i[:, :1], vocab_size, dtype=logits_indices_i.dtype)], dim=-1)
                    logits_values.append(logits_values_i)
                    logits_indices.append(logits_indices_i)
                    break
                k += 1000
        
        # Since k might be different for each position, we need to handle this differently
        # Instead of stacking, we'll create COO indices directly
        
        # Create lists to hold all indices and values
        all_batch_indices = []
        all_seq_indices = []
        all_vocab_indices = []
        all_values = []
        
        # For each position in the sequence
        for seq_idx, (values_i, indices_i) in enumerate(zip(logits_values, logits_indices)):
            # For each sample in the batch
            for batch_idx in range(n_samples):
                # Get number of values for this specific (batch, seq) pair
                num_values = values_i[batch_idx].size(0)
                
                # Add batch indices
                all_batch_indices.append(torch.full((num_values,), batch_idx, device=device))
                
                # Add sequence indices
                all_seq_indices.append(torch.full((num_values,), seq_idx, device=device))
                
                # Add vocabulary indices
                all_vocab_indices.append(indices_i[batch_idx])
                
                # Add values
                all_values.append(values_i[batch_idx])
        
        # Concatenate all indices and values
        batch_indices = torch.cat(all_batch_indices)
        seq_indices = torch.cat(all_seq_indices)
        vocab_indices = torch.cat(all_vocab_indices)
        values = torch.cat(all_values)
        
        # Create the indices tensor for the sparse COO tensor
        indices = torch.stack([
            batch_indices,  # batch dimension
            seq_indices,    # sequence dimension
            vocab_indices   # vocabulary dimension
        ], dim=0)

        # Create sparse tensor
        sparse_logits = torch.sparse_coo_tensor(
            indices=indices,
            # float() is necessary here as sparse.softmax is not implemented for bf16.
            values=values.float(),
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
        return tu.cpu().tolist(), au.cpu().tolist(), (tu - au).cpu().tolist()

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
    mean_logits = torch.log(F.softmax(logits, dim=-1).mean(dim=0)) # shape: [seq_len, vocab_size]
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

def evaluate_uncertainty_all_mem_efficient(logits, token_ids, mode="default"):
    """
    Uncertainty evaluation for the logits.
    Args:
        logits: [n_samples, seq_len, vocab_size]
        token_ids: [seq_len, 1]
    Returns: uncertainties of the shape: [seq_len]
        TUs: total uncertainty, 
            H(E_{\theta}[P(y|x, \theta)])
        AUs: aleotoric uncertainty, 
            E_{\theta}[H(P(y|x, \theta))]
        EUs: epistemic uncertainty, 
            TU - AU
        BIRs: Bayesian Implicit Reward, 
            E_{\theta}[log(E_{\theta}[P(y|x, \theta)]) - log(P(y|x, \theta))]
    """
    probs = softmax_inplace(logits)
    log_probs_buffer = torch.zeros_like(probs[:, 0, :])

    ##########################################
    ### Uncertainties
    ##########################################
    tus, aus, eus = [], [], []
    for i in range(probs.size(1)):
        # the following operations are all in-place,
        # to save memory.
        log_probs_buffer.copy_(probs[:, i, :])
        log_probs_buffer.log_()
        log_probs_buffer.mul_(probs[:, i, :])
        au = -log_probs_buffer.sum(dim=-1).mean().item()

        mean_probs = probs[:, i, :].mean(dim=0)
        log_probs_buffer[0].copy_(mean_probs)
        log_probs_buffer[0].log_()
        log_probs_buffer[0].mul_(mean_probs)
        tu = -log_probs_buffer[0].sum().item()

        eu = tu - au
        
        tus.append(tu)
        aus.append(au)
        eus.append(eu)

    ##########################################
    ### Bayesian Implicit Reward
    ##########################################
    tensor_token_ids = torch.tensor(token_ids, device=probs.device).squeeze(-1) # shape: [seq_len,]
    # print("shape of token_ids: ", tensor_token_ids.shape)
    # print("sampled token_ids: ", tensor_token_ids)
    # print("shape of probs: ", probs.shape)
    
    truncated_probs = probs[:,-tensor_token_ids.shape[0]:,]
    # print("argmax of the logits: ", truncated_probs[0].argmax(dim=-1))
    
    # get the log probs of the sampled tokens
    # shape: [n_samples, seq_len]
    sampled_probs = truncated_probs.gather(
        dim=2, 
        index=tensor_token_ids.tile(truncated_probs.shape[0], 1).unsqueeze(-1),
    ).squeeze(-1)

    # compute the Bayesian Implicit Reward.
    #   "default": E_theta[log(E_theta[P(y|x, theta)]) - log(P(y|x, theta))]
    #   "best-vs-random": the first sample is the best sample without noise.
    if mode == "default":
        r_win = torch.log(sampled_probs.mean(dim=0))
    elif mode == "best-vs-random":
        r_win = torch.log(sampled_probs[0])
    else: 
        raise ValueError(f"Unknown mode for Bayesian Implicit Reward Estimation: {mode}.")
    
    # r_lose is the average log probs of the sampled tokens.
    r_lose = torch.log(sampled_probs).mean(dim=0)

    # starting 0s represent the tokens of the prompt. 
    birs = [0. for _ in range(len(tus) - r_lose.shape[0])] + (r_win - r_lose).cpu().tolist()
    
    return tus, aus, eus, birs