import torch as th
import torch.nn as nn
import blobfile as bf
from tqdm import tqdm
import subprocess
import os

from .utils import distribution_registry

MODEL_NAMES = ["gelu-1l", "gelu-2l", "gelu-4l"]
DISTRIB_NAMES = ["hex", "camel", "indent", "ifelse", "caps", "english", "spanish", "icl"]
N_REPS_DICT = {"hex": 32, "indent": 32, "ifelse": 32, "camel": 32, "caps": 16, "english": 24, "spanish": 24, "icl": 1}
RECOMMENDED_TEMPS = {
    "gelu-1l": {
        "ITGIS": {"hex": 1.0, "camel": 1.0, "indent": 1.0, "ifelse": 1.0, "caps": 1.5, "english": 0.45, "spanish": 0.67, "icl": 0.45},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.0, "ifelse": 2.24, "caps": 3.34, "english": 1.5, "spanish": 2.24, "icl": 1.0}
    },
    "gelu-2l": {
        "ITGIS": {"hex": 1.5, "camel": 1.5, "indent": 1.0, "ifelse": 0.45, "caps": 0.45, "english": 0.67, "spanish": 0.67, "icl": 0.3},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.5, "ifelse": 1.5, "caps": 1.5, "english": 2.24, "spanish": 2.24, "icl": 0.67}
    },
    "gelu-4l": {
        "ITGIS": {"hex": 5.0, "camel": 1.0, "indent": 0.67, "ifelse": 1.0, "caps": 0.67, "english": 0.45, "spanish": 1.0, "icl": 3.34},
        "MHIS": {"hex": 0.67, "camel": 2.24, "indent": 1.0, "ifelse": 1.0, "caps": 1.0, "english": 1.5, "spanish": 2.24, "icl": 0.67}
    }
}

class FactorGaussian:
    def __init__(self, mean, L_factor, diagonal):
        """Σ = L @ L.T + diag(diagonal)"""
        self.mean = mean
        self.L = L_factor
        self.diagonal = diagonal

    def covariance(self):
        stable_diagonal = self.diagonal + 1e-8
        return self.L @ self.L.T + th.diag(stable_diagonal)

    def get_diagonal(self):
        l_diag = th.sum(self.L * self.L, dim=1)
        return l_diag + self.diagonal

def get_gse_grouped_layers(model):
    """Groups the final layers of the model for the Gaussian Sampling Estimator."""
    return [nn.Sequential(model.ln_final, model.unembed)]

def propagate_by_sampling(input_distribution, layer_block, mask=None, num_samples=1000, device='cpu'):
    input_mean = input_distribution.mean
    input_cov = input_distribution.covariance()
    input_dim = input_mean.shape[0]

    epsilon = 1e-6 * th.eye(input_dim, device=input_mean.device)
    input_dist_sampler = th.distributions.MultivariateNormal(loc=input_mean, covariance_matrix=input_cov + epsilon)
    samples = input_dist_sampler.sample(th.Size([num_samples]))

    samples_reshaped = samples.unsqueeze(1)

    with th.no_grad():
        output_samples_reshaped = layer_block(samples_reshaped.to(device))

    output_samples = output_samples_reshaped.squeeze(1)

    output_mean = th.mean(output_samples, dim=0)
    mean_subtracted_output = output_samples - output_mean
    full_output_covariance = (1 / (num_samples - 1)) * mean_subtracted_output.T @ mean_subtracted_output
    output_dim = output_mean.shape[0]

    if mask is not None and len(mask) > 0:
        masked_cov = th.zeros_like(full_output_covariance)
        valid_mask_entries = [(r, c) for r, c in mask if r < output_dim and c < output_dim]
        for r, c in valid_mask_entries:
            masked_cov[r, c] = full_output_covariance[r, c]
            if r != c:
                masked_cov[c, r] = full_output_covariance[c, r]
    else:
        masked_cov = th.diag(th.diag(full_output_covariance))

    output_diag_variances = th.diag(full_output_covariance)
    off_diag_cov = masked_cov - th.diag(th.diag(masked_cov))

    try:
        eigvals, eigvecs = th.linalg.eigh(off_diag_cov)
        eigvals_positive = th.clamp(eigvals, min=0)
        sqrt_eigvals = th.sqrt(eigvals_positive)
        tol = 1e-6
        rank = th.sum(eigvals_positive > tol)
        output_L = eigvecs[:, -rank:] @ th.diag(sqrt_eigvals[-rank:])
        l_diag_contribution = th.sum(output_L * output_L, dim=1)
        output_diag_remainder = th.clamp(output_diag_variances - l_diag_contribution, min=0)
    except th.linalg.LinAlgError:
        output_L = th.zeros((output_dim, 1), device=output_mean.device)
        output_diag_remainder = output_diag_variances

    return FactorGaussian(mean=output_mean, L_factor=output_L, diagonal=output_diag_remainder)


def compute_estimator_kl_div(true_output_distribution, predicted_output_distribution):
    mu_true, mu_pred = true_output_distribution.mean, predicted_output_distribution.mean
    sigma_true = true_output_distribution.covariance().detach().cpu().numpy()
    sigma_pred = predicted_output_distribution.covariance().detach().cpu().numpy()
    mu_true, mu_pred = mu_true.detach().cpu().numpy(), mu_pred.detach().cpu().numpy()

    epsilon = 1e-6 * np.eye(sigma_true.shape[0])
    sigma_true_stable, sigma_pred_stable = sigma_true + epsilon, sigma_pred + epsilon

    mu_diff = mu_pred - mu_true
    k = len(mu_true)
    try:
        sigma_pred_inv = np.linalg.inv(sigma_pred_stable)
        trace_term = np.trace(sigma_pred_inv @ sigma_true_stable)
        quad_term = mu_diff.T @ sigma_pred_inv @ mu_diff
        sign_true, logdet_true = np.linalg.slogdet(sigma_true_stable)
        sign_pred, logdet_pred = np.linalg.slogdet(sigma_pred_stable)
        if sign_true <= 0 or sign_pred <= 0: return float('inf')
        log_det_term = logdet_pred - logdet_true
        return 0.5 * (trace_term + quad_term - k + log_det_term)
    except np.linalg.LinAlgError:
        return float('inf')


def load_ground_truth(model_name: str, dist_names: list[str] = DISTRIB_NAMES, device: str = "cpu"):
    """
    Returns a dictionary of tensors of length `vocab_size` denoting the frequencies. The sum of each tensor is 2^32.
    """
    assert model_name in MODEL_NAMES
    gt_freqs = {}
    
    for dist_name in dist_names:
        with bf.BlobFile(f"gs://arc-ml-public/lpe/ground-truth/{model_name}/frequencies32-{dist_name}.pt"
, "rb") as f:
            gt_freqs[dist_name] = th.load(f, map_location=device, weights_only=True)
        
        assert th.sum(gt_freqs[dist_name]).item() == 2**32
    
    return gt_freqs

def gen_activ_samples(model, dist_name: str, n_samples: int, batch_size: int = 64, show_progress: bool = False):
    """
    Generates `n_samples` samples of the pre-unembed activations from the distribution `dist_name`.
    """
    assert dist_name in DISTRIB_NAMES
    dist = distribution_registry[dist_name](model.tokenizer)
    samples = dist.sample(n_reps=N_REPS_DICT[dist_name], n_samples=n_samples)
    acts = []
    with th.no_grad():
        for batch in tqdm(samples.split(batch_size, dim=0)):
            x = model.embed(batch) + model.pos_embed(batch)
            for block in model.blocks:
                x = block(x)
            acts.append(x[:,-1,:].clone())
    return th.cat(acts)

def pick_random_tokens(gt: th.Tensor, count, p_min=1e-9, p_max=1e-5):
    """
    Returns `count` random tokens whose ground-truth probabilities are in the range [p_min, p_max].
    """
    gt_probs = gt / gt.sum()
    valid_idx = th.logical_and(gt_probs >= p_min, gt_probs <= p_max).nonzero().squeeze()
    sampled_idx = th.randperm(valid_idx.size(0))[:count]
    return valid_idx[sampled_idx].tolist()