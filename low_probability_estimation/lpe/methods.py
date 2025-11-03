import torch as th
import torch
from fancy_einsum import einsum
from tqdm import tqdm
import numpy as np
from torch.distributions import MultivariateNormal


from .utils import Discrete


def random_constraint_projection(
    P: th.tensor, biases: th.tensor, n_reps=1000, tol=1e-9, factor=0.99
):
    """
    Finds shortest point in the polytope defined by P @ x + biases >= 0.
    This method isn't very airtight, but it was what we used in our experiments.
    """
    n, d = P.shape
    device = P.device

    x = th.zeros(d, device=device)

    for _ in range(n_reps * 100):
        constraints = P @ x + biases
        # This should be be -tol, but we used tol in the official experiments
        unsatisfied = constraints < tol

        # Randomly choose an unsatisfied constraint
        # This will crash if there are no unsatisfied constraints?
        unsatisfied_indices = th.where(unsatisfied)[0]
        if len(unsatisfied_indices) == 0:
            if _ < n_reps:
                x *= factor
            else:
                return x
        else:
            random_index = unsatisfied_indices[th.randint(len(unsatisfied_indices), (1,))]

            # Project onto the chosen constraint
            p_i = P[random_index]

            projection = ((-constraints[random_index]) / th.sum(p_i**2)) * p_i
            x += projection.squeeze()

    assert False, "Did not find a feasible point"


def QLD(
    W_U: th.Tensor, act_samps: th.Tensor, target: int, *, batch_size: int = 512
) -> float:
    """
    Quadratic Logit Decomposition. Estimates the probability of outputing token `target` by using the Quadratic Logit Decomposition with the Shortest Accepting Vector as `d'.
    Inputs:
    - W_U: the unembedding matrix of the model (d_model x d_vocab).
    - act_samps: the samples of the activations right before unembedding (n_samples x d_model).
    - target: the target token (in range [0...d_model)).
    Returns:
    - The estimated probability of outputing token `target`.
    """

    d_model = W_U.shape[0]
    d_vocab = W_U.shape[1]
    n = act_samps.shape[0]
    assert target < d_vocab, "Target token out of range"
    assert act_samps.shape == (n, d_model), "act_samps has incorrect shape"

    # whiten the activations.
    n = act_samps.shape[0]
    mean = act_samps.mean(dim=0)
    cov = (act_samps - mean).T @ (act_samps - mean) / n
    EPS = 1e-5
    A = th.linalg.cholesky(
        cov + EPS * th.eye(d_model, device=cov.device)
    )  # (d_model, d_model). We have cov + EPS*I == A @ A.T.
    u_samps = act_samps - mean @ th.inverse(A.T)  # (n, d_model), the whitened samples.

    # Given whitened samples z of shape (n, d_model), the logits are given by z @ pullbacks.T + biases. Every row of the logits is a sample.
    pullbacks = W_U.T @ A  # (d_vocab, d_model)
    biases = W_U.T @ mean  # (d_vocab, ).

    # find the shortest accepting vector.
    pullbacks_diff = pullbacks[target].unsqueeze(0) - pullbacks
    biases_diff = biases[target].unsqueeze(0) - biases
    biases_diff[target] = 100
    d = random_constraint_projection(
        pullbacks_diff, biases_diff, n_reps=200, factor=0.95
    )  # (d_model, )
    d = d / th.norm(d)

    a_samps = (u_samps @ d).sort().values
    all_probs = []

    assert n % batch_size == 0
    for y in th.split(u_samps, batch_size):
        b_samps = y - y @ th.outer(d, d)  # (batch_size, d_model)

        # figure out the lower and upper bounds on the last direction.
        # Let z be b_samps. For a particular sample i, we need that for all j != t=token, we have:
        #  z_i @ pullbacks[t].T + biases[t] > z_i @ pullbacks[j].T + biases[j].
        # If we let z_i = a_i + r_i d, where r_i is a scalar, then we need that:
        #  a_i @ (pullbacks[t] - pullbacks[j]).T + biases[t] - biases[j] > -r_i d @ (pullbacks[t] - pullbacks[j]).T
        #  {a_i @ (pullbacks[t] - pullbacks[j]).T + biases[t] - biases[j]} / -{d @ (pullbacks[t] - pullbacks[j]).T} > r_i   (possibly with a sign flip)

        pullbacks_diff = (pullbacks[target].unsqueeze(0) - pullbacks).mT
        numerator = y @ pullbacks_diff + biases[target] - biases
        denominator = -d @ pullbacks_diff
        lower = (
            th.where(denominator < 0, numerator / denominator, -th.inf).max(-1).values
        )  # (batch_size, )
        upper = (
            th.where(denominator > 0, numerator / denominator, th.inf).min(-1).values
        )  # (batch_size, )

        # find how many latents were between upper and lower
        all_probs.append(
            th.maximum(
                th.searchsorted(a_samps, upper) - th.searchsorted(a_samps, lower),
                th.tensor(0),
            )
            / n  # (batch_size, )
        )

    all_probs = th.cat(all_probs)  # (n, )
    return all_probs.mean().item()


def GLD(
    W_U: th.Tensor, act_samps: th.Tensor, target: int, *, batch_size: int = 512
) -> float:
    """
    Gaussian Logit Difference. Finds parameters of the normal distribution fit to the target logit minus the maximum logit.
    Inputs:
    - W_U: the unembedding matrix of the model (d_model x d_vocab).
    - act_samps: the samples of the activations right before unembedding (n_samples x d_model).
    - target: the target token (in range [0...d_model)).
    Returns:
    - mu, sigma: The mean and variance of the logit differnce. Note mu <= 0.
    """

    argmax = []
    # Use batches to avoid OOM
    for batch in act_samps.split(batch_size, dim=0):
        logits = batch @ W_U
        argmax.append(logits.argmax(dim=1))
    argmax = th.cat(argmax)

    max_samps = einsum("b x, x b -> b", act_samps, W_U[:, argmax])
    target_samps = act_samps @ W_U[:, target]

    mu = (target_samps - max_samps).mean().item()
    sigma = (target_samps - max_samps).std().item()

    return mu, sigma


def ITGIS(
    model,
    orig_dists: list[Discrete],
    target: int,
    *,
    temp: float,
    n_samples: int,
    batch_size: int = 256,
    decay_rate: float = 0.9,
    show_progress: bool = False
) -> float:
    """
    Independent Token Gradient Importance Sampling. Uses the gradient of the logit with respect to the token embedding to define a new importance sampling distribution (with all tokens still being independent). Adaptively updates the importance sampling distribution based on samples from the previous.
    Inputs:
    - model: the transformer.
    - orig_dists: list of Discrete distributions for each token position.
    - target: the target token (in range [0...d_vocab)).
    - temp: the temperature.
    - n_samples: the number of samples to be drawn.
    - batch_size: the batch size.
    - decay_rate: the decay rate in the exponentially-weighted moving average of gradients.
    Returns:
    - The estimated probability of outputing token `target`.
    """

    d_vocab = model.embed.d_vocab
    ctx_len = len(orig_dists)
    scores = th.zeros((ctx_len, d_vocab), device=model.device)

    for param in model.parameters():
        param.requires_grad_(False)

    imp_samp_probs = []
    assert n_samples % batch_size == 0
    for i in tqdm(list(range(n_samples // batch_size)), disable=not show_progress):
        target_samples = []
        target_logratios = []
        adj_temp = (
            temp * (1 - decay_rate**i) / (1 - decay_rate) if i > 0 else 1
        )  # adjust temperature for exponential moving average
        for dist, scores_at_pos in zip(orig_dists, scores):
            samples_at_pos = dist.boltzmann_distribution(
                scores=scores_at_pos[dist.values], temperature=adj_temp
            ).sample((batch_size,))
            target_samples.append(samples_at_pos)
            target_logratios.append(
                th.logsumexp(
                    scores_at_pos[dist.values] / adj_temp + th.log(dist.probs), dim=0
                )
                - scores_at_pos[samples_at_pos] / adj_temp
            )

        samples = th.stack(target_samples, dim=1)  # (batch_size, ctx_len)
        logratios = th.stack(target_logratios, dim=1)  # (batch_size, ctx_len)

        with th.enable_grad():
            onehot = th.nn.functional.one_hot(
                samples, num_classes=d_vocab
            ).float()  # (batch_size, ctx_len, d_vocab)
            onehot.requires_grad_(True)

            x = (
                onehot @ model.embed.W_E
            )  # (batch_size, context_len, d_model) @ (d_model, d_vocab) -> (batch_size, context_len, d_vocab)
            x = x + model.pos_embed(samples)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x)[:, -1, :]
            y = model.unembed(x[:, None, :])  # (batch_size, 1, d_vocab)

            probs = (y[:, -1, :].argmax(-1) == target).float().detach()
            imp_samp_probs.append((th.exp(logratios.sum(-1)) * probs))

            (
                x @ model.unembed.W_U[:, target]
            ).sum().backward()  # for some reason, this is faster than y[:,-1,target].sum().backward():
            scores *= decay_rate
            scores += onehot.grad.sum(0) / batch_size

    imp_samp_probs = th.cat(imp_samp_probs, dim=0)
    return imp_samp_probs.mean().item()

def MHIS(
    model,
    orig_dists: list[Discrete],
    target: int,
    *,
    temp: float,
    n_samples: int,
    burn_in: int,
    batch_size: int = 32,
    show_progress: bool = False
) -> float:
    """
    Metropolis-Hastings Importance Sampling. Takes batch_size independent random walks in token space, with q(x) \propto exp(logit / temp) * p(x) as the stationary distribution.
    We use the proposal function phi(x'|x) defined by:
    - Choose a random token position i
    - Take the gradient of s(x) with respect to the embedding of token i. Dot that with the different tokens y you could replace token i with, and take probabilities proportional to p(y) exp(grad \cdot y / temp).
    Inputs:
    - model: the transformer.
    - orig_dists: list of Discrete distributions for each token position.
    - target: the target token (in range [0...d_vocab)).
    - temp: the temperature (for both the Boltzmann stationary distribution and proposal distribution)
    - n_samples: the total number of samples to be drawn, ignoring burn-in.
    - batch_size: the number of parallel random walks to run (the total number of samples drawn is n_samples + burn_in * batch_size)
    Returns:
    - The estimated probability of outputing token `target`.
    """
    d_vocab = model.embed.d_vocab
    ctx_len = len(orig_dists)
    scores = th.zeros((ctx_len, d_vocab), device=model.device)

    for param in model.parameters():
        param.requires_grad_(False)

    orig_log_probs = []
    for pos in range(ctx_len):
        mask = -th.inf * th.ones(d_vocab, device=model.device)
        mask[orig_dists[pos].values] = th.log(orig_dists[pos].probs)
        orig_log_probs.append(mask)
    orig_log_probs = th.stack(orig_log_probs)

    for param in model.parameters():
        param.requires_grad_(False)

    results = []
    scores = []

    with th.enable_grad():
        # Initialize the first batch of samples
        current_samples = th.stack([dist.sample((batch_size,)) for dist in orig_dists], dim=1)

        acceptance_rate = 0
        total_proposals = 0

        for step in tqdm(range((n_samples // batch_size + burn_in)), disable=not show_progress):

            if step == 0:

                onehot = th.nn.functional.one_hot(current_samples, num_classes=d_vocab).float()
                onehot.requires_grad_(True)
                x = onehot @ model.embed.W_E
                x = x + model.pos_embed(current_samples)
                for block in model.blocks:
                    x = block(x)
                x = model.ln_final(x[:,-1].unsqueeze(1))
                y = model.unembed(x).squeeze(1)
                current_scores = y[:, target]
                current_scores.sum().backward()
                current_scores = current_scores.detach().clone()
                current_grads = onehot.grad.detach().clone()
                current_results = (y.argmax(dim=-1) == target).float()

            pos = th.randint(0, ctx_len, (batch_size,), device=current_samples.device)
            
            # Compute proposal probabilities
            proposal_logits = orig_log_probs[pos] + current_grads[th.arange(batch_size), pos] / temp
            proposal_probs = th.softmax(proposal_logits, dim=-1)
            
            # Propose new tokens
            proposed_tokens = th.multinomial(proposal_probs, 1).squeeze(-1)

            # Create proposed samples
            proposed_samples = current_samples.clone()
            proposed_samples[th.arange(batch_size), pos] = proposed_tokens

            # Recompute scores and gradients for proposed samples
            onehot = th.nn.functional.one_hot(proposed_samples, num_classes=d_vocab).float()
            onehot.requires_grad_(True)
            x = onehot @ model.embed.W_E
            x = x + model.pos_embed(proposed_samples)
            for block in model.blocks:
                x = block(x)
            x = model.ln_final(x[:,-1].unsqueeze(1))
            y = model.unembed(x).squeeze(1)
            proposed_scores = y[:, target]
            proposed_scores.sum().backward()
            proposed_grads = onehot.grad.clone()
            proposed_results = (y.argmax(dim=-1) == target).float()

            # Clear gradients
            onehot.grad = None

            # Compute reverse proposal probabilities
            reverse_proposal_logits = orig_log_probs[pos] + proposed_grads[th.arange(batch_size), pos] / temp
            reverse_proposal_probs = th.softmax(reverse_proposal_logits, dim=-1)

            # Compute log acceptance probabilities
            log_accept_probs = (proposed_scores - current_scores) / temp + \
                               orig_log_probs[pos, proposed_tokens] - orig_log_probs[pos, current_samples[th.arange(batch_size), pos]] + \
                               th.log(reverse_proposal_probs[th.arange(batch_size), current_samples[th.arange(batch_size), pos]]) - \
                               th.log(proposal_probs[th.arange(batch_size), proposed_tokens])

            # Accept or reject proposals
            accept_mask = th.log(th.rand(batch_size, device=log_accept_probs.device)) < log_accept_probs
            current_samples[accept_mask] = proposed_samples[accept_mask]
            current_scores[accept_mask] = proposed_scores[accept_mask]
            current_grads[accept_mask] = proposed_grads[accept_mask]
            current_results[accept_mask] = proposed_results[accept_mask]

            current_scores = current_scores.detach().clone()
            current_grads = current_grads.detach().clone()
            current_results = current_results.detach().clone()
            current_samples = current_samples.detach().clone()

            if step >= burn_in:
                results.append(current_results.detach().clone())
                scores.append(current_scores.detach().clone())

            acceptance_rate += accept_mask.float().mean().item()
            total_proposals += 1

    acceptance_rate /= total_proposals

    results = th.cat(results)
    scores = th.cat(scores)
    exp_scores = th.exp(scores / temp)
    normalizing_constant = 1 / (1 / exp_scores).mean().item()  # E_p[exp(s(x)/temp)]
    unbiased_estimates = results * normalizing_constant / exp_scores

    return unbiased_estimates.mean().item()


# --- Start of code from ReluNN_gaussian_sampling_estimator.ipynb ---

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

def propagate_by_sampling(input_distribution, layer_block, mask=None, num_samples=1000, device='cpu'):
    input_mean = input_distribution.mean
    input_cov = input_distribution.covariance()
    input_dim = input_mean.shape[0]
    
    print(f"    - Propagating through layer: {layer_block.__class__.__name__}")
    print(f"    - Input mean shape: {input_mean.shape}")
    print(f"    - Input covariance shape: {input_cov.shape}")
    print(f"    - Number of samples for propagation: {num_samples}")
    
    epsilon = 1e-6 * th.eye(input_dim, device=input_mean.device)
    try:
        input_dist_sampler = MultivariateNormal(loc=input_mean, covariance_matrix=input_cov + epsilon)
    except torch.linalg.LinAlgError:
        print("    - WARNING: Covariance matrix not positive definite. Adding larger epsilon.")
        epsilon = 1e-4 * torch.eye(input_dim, device=input_mean.device)
        input_dist_sampler = MultivariateNormal(loc=input_mean, covariance_matrix=input_cov + epsilon)

    samples = input_dist_sampler.sample(th.Size([num_samples]))
    print(f"    - Drawn samples shape: {samples.shape}")
    
    samples_reshaped = samples.unsqueeze(1)

    with th.no_grad():
        output_samples_reshaped = layer_block(samples_reshaped.to(device))

    output_samples = output_samples_reshaped.squeeze(1)
    print(f"    - Output samples shape after layer block: {output_samples.shape}")

    output_mean = th.mean(output_samples, dim=0)
    mean_subtracted_output = output_samples - output_mean
    full_output_covariance = (1 / (num_samples - 1)) * mean_subtracted_output.T @ mean_subtracted_output
    output_dim = output_mean.shape[0]
    
    print(f"    - Output mean shape: {output_mean.shape}")
    print(f"    - Full output covariance shape: {full_output_covariance.shape}")

    if mask is not None and len(mask) > 0:
        print(f"    - Applying mask with {len(mask)} entries.")
        masked_cov = th.zeros_like(full_output_covariance)
        valid_mask_entries = [(r, c) for r, c in mask if r < output_dim and c < output_dim]
        rows, cols = zip(*valid_mask_entries)
        masked_cov[rows, cols] = full_output_covariance[rows, cols]
    else:
        print("    - No mask provided, using diagonal covariance.")
        masked_cov = th.diag(th.diag(full_output_covariance))
        
    output_diag_variances = th.diag(full_output_covariance)
    off_diag_cov = masked_cov - th.diag(th.diag(masked_cov))

    try:
        eigvals, eigvecs = th.linalg.eigh(off_diag_cov)
        eigvals_positive = th.clamp(eigvals, min=0)
        sqrt_eigvals = th.sqrt(eigvals_positive)
        tol = 1e-6
        rank = th.sum(eigvals_positive > tol).item()
        print(f"    - Off-diagonal covariance matrix rank: {rank}")
        
        if rank > 0:
            output_L = eigvecs[:, -rank:] @ th.diag(sqrt_eigvals[-rank:])
            l_diag_contribution = th.sum(output_L * output_L, dim=1)
            output_diag_remainder = th.clamp(output_diag_variances - l_diag_contribution, min=0)
        else:
            output_L = th.zeros((output_dim, 1), device=output_mean.device)
            output_diag_remainder = output_diag_variances
            print("    - Rank is 0, using diagonal remainder for covariance.")

    except torch.linalg.LinAlgError:
        print("    - WARNING: Eigendecomposition failed. Falling back to diagonal approximation.")
        output_L = th.zeros((output_dim, 1), device=output_mean.device)
        output_diag_remainder = output_diag_variances

    print(f"    - L factor shape: {output_L.shape}")
    print(f"    - Diagonal remainder shape: {output_diag_remainder.shape}")

    return FactorGaussian(mean=output_mean, L_factor=output_L, diagonal=output_diag_remainder)


def gaussian_sampling_estimator(model, orig_dists: list[Discrete], target: int, *, n_samples: int, batch_size: int, n_off_diagonal_entries: int = 0, show_progress: bool = False) -> float:
    """
    """
    # Run the initial samples through the first layer to create the initial distribution we'll draw from
    initial_samples = th.stack([dist.sample((n_samples,)) for dist in orig_dists], dim=1)
    with th.no_grad():
        tok_emb = model.embed(initial_samples)
        pos_emb = model.pos_embed(initial_samples)
        initial_activations = tok_emb + pos_emb
        flat_initial_activations = initial_activations.reshape(-1, model.cfg['d_model'])
    mean = th.mean(flat_initial_activations, dim=0)
    cov = th.cov(flat_initial_activations.T)
    epsilon_diag = 1e-6 * th.eye(cov.shape[0], device=cov.device)
    # Compute Cholesky decomposition for initial covariance (very helpful when we start adding off-diagonal entries)
    L = th.linalg.cholesky(cov + epsilon_diag)
    diag = th.zeros_like(mean) 
    current_dist = FactorGaussian(mean=mean, L_factor=L, diagonal=diag)

    # Propagate with no masking
    print(f"MODEL BLOCKS: {model.blocks}")
    grouped_layers = model.blocks
    masks = []
    all_covariance_entries = []
    temp_dist = current_dist
    for layer_idx, layer_block in enumerate(grouped_layers):
        output_dim = model.cfg['d_model']
        print(f"Output Dim: {output_dim}")
        full_mask = set((r, c) for r in range(output_dim) for c in range(output_dim))
        temp_dist = propagate_by_sampling(temp_dist, layer_block, mask=full_mask, num_samples=10000, device=model.device)
        full_cov = temp_dist.covariance()
        for r in range(output_dim):
            for c in range(r + 1, output_dim):
                all_covariance_entries.append((abs(full_cov[r, c].item()), layer_idx, r, c))
    all_covariance_entries.sort(key=lambda x: x[0], reverse=True)

    # Propagate with masking (just for fun)
    for block in grouped_layers:
        dim = model.cfg['d_model']
        masks.append(set((i, i) for i in range(dim)))
    for _, layer_idx, r, c in all_covariance_entries[:n_off_diagonal_entries]:
        masks[layer_idx].add((r,c))
        masks[layer_idx].add((c,r))
    for i, layer_block in enumerate(grouped_layers):
        current_dist = propagate_by_sampling(current_dist, layer_block, mask=masks[i], num_samples=batch_size, device=model.device)
    # Propagate through final layer norm
    final_ln_mask = set((i, i) for i in range(model.cfg['d_model']))
    current_dist = propagate_by_sampling(current_dist, model.ln_final, mask=final_ln_mask, num_samples=batch_size, device=model.device)

    # 3. Final probability estimation
    final_mean = current_dist.mean
    final_cov = current_dist.covariance()
    final_dist_sampler = MultivariateNormal(loc=final_mean, covariance_matrix=final_cov + epsilon_diag)
    final_act_samples = final_dist_sampler.sample(th.Size([n_samples]))
    print(f"  - Sampled final activations, shape: {final_act_samples.shape}")
    
    # Get logits and probabilities
    with th.no_grad():
        logits = final_act_samples @ model.unembed.W_U
        probs = th.softmax(logits, dim=-1)
    
    print(f"  - Computed final logits and probabilities, shape: {probs.shape}")

    # Average probability of the target token
    target_probs = probs[:, target]
    estimated_prob = target_probs.mean().item()
    print(f"\n--- Estimated Probability for target {target}: {estimated_prob:.6e} ---")
    
    return estimated_prob


# --- End of code from ReluNN_gaussian_sampling_estimator.ipynb ---