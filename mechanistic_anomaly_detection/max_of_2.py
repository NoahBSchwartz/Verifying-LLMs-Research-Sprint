import torch as th
from collections import defaultdict
import torch.nn.functional as F
import numpy as np

def get_tensor_stats(tensor, name):
    """Calculate comprehensive statistics for a tensor."""
    with th.no_grad():
        stats = {
            'name': name,
            'shape': tuple(tensor.shape),
            'mean': float(tensor.mean()),
            'std': float(tensor.std()),
            'min': float(tensor.min()),
            'max': float(tensor.max()),
            'norm': float(tensor.norm()),
            'sparsity': float((tensor == 0).float().mean()),
            'abs_mean': float(tensor.abs().mean())
        }
        
        # Add condition number for 2D matrices
        if len(tensor.shape) == 2:
            try:
                singular_values = th.linalg.svd(tensor)[1]
                stats['condition_number'] = float(singular_values[0] / singular_values[-1])
            except:
                stats['condition_number'] = float('inf')
                
        return stats

def analyze_activation_patterns(activation, name):
    """Analyze activation patterns during forward pass."""
    with th.no_grad():
        stats = get_tensor_stats(activation, name)
        
        # Additional activation-specific metrics
        if len(activation.shape) >= 2:
            stats.update({
                'dead_neurons': float((activation.mean(0) == 0).float().mean()),
                'saturation': float((activation.abs() > 0.99).float().mean())
            })
            
        return stats

def masked_softmax(logits):
    n_ctx = logits.shape[-1]
    mask = th.tril(th.ones(n_ctx, n_ctx), 0)
    mask = mask[(None,) * (logits.ndim - mask.ndim) + (Ellipsis,)]
    mask = mask.to(logits.device)
    logits = logits - logits.max(dim=-1, keepdim=True)[0]
    probs = th.exp(logits) * mask
    probs = probs / probs.sum(dim=-1, keepdim=True)
    return probs

def forward(model, input, *, hook_fn=None, collect_stats=False):
    activation_stats = defaultdict(list) if collect_stats else None
    
    def hook(key, x):
        if collect_stats:
            activation_stats[key].append(analyze_activation_patterns(x.detach(), key))
        if hook_fn is None:
            return x
        new_x = hook_fn(key, x)
        return x if new_x is None else new_x
    
    if th.is_floating_point(input):
        x = input[(None,) * (3 - input.ndim)]
    else:
        x = input[(None,) * (2 - input.ndim)]
        x = th.nn.functional.one_hot(x, num_classes=model.cfg.d_vocab).float()
    
    x = x.to(model.W_E.device)
    x = x @ model.W_E
    x = hook("embed", x)
    x = x + model.W_pos[None, : x.shape[-2]]
    x = hook("attn_block_pre", x)
    
    Q = x[:, None] @ model.W_Q[None, 0] + model.b_Q[None, 0].unsqueeze(-2)
    K = x[:, None] @ model.W_K[None, 0] + model.b_K[None, 0].unsqueeze(-2)
    V = x[:, None] @ model.W_V[None, 0] + model.b_V[None, 0].unsqueeze(-2)
    Q = hook("Q", Q)
    K = hook("K", K)
    V = hook("V", V)
    
    d_k = model.W_K.shape[-1]
    attn_prob = masked_softmax(Q @ K.transpose(-2, -1) / d_k**0.5)
    attn_prob = hook("attn_prob", attn_prob)
    attn = attn_prob @ V
    attn = hook("attn_z", attn)
    attn = (attn @ model.W_O[None, 0]).sum(-3) + model.b_O[None, 0].unsqueeze(-2)
    attn = hook("attn_result", attn)
    x = x + attn
    
    x = hook("attn_block_post", x)
    x = x @ model.W_U + model.b_U[None, None]
    
    return x, activation_stats if collect_stats else x

def analyze_model_errors(model, forward_fn, all_sequences):
    """
    Analyzes and prints out cases where the model makes incorrect predictions.
    
    Args:
        model: The transformer model
        forward_fn: The forward function to use
        all_sequences: Tensor of all possible input sequences
    
    Returns:
        tuple: (error_cases, accuracy, soft_accuracy, network_stats)
    """
    network_stats = {
        'parameters': {},
        'activations': defaultdict(list)
    }
    
    # Analyze model parameters
    for name, param in [
        ('W_E', model.W_E),
        ('W_pos', model.W_pos),
        ('W_Q', model.W_Q),
        ('W_K', model.W_K),
        ('W_V', model.W_V),
        ('W_O', model.W_O),
        ('W_U', model.W_U),
        ('b_Q', model.b_Q),
        ('b_K', model.b_K),
        ('b_V', model.b_V),
        ('b_O', model.b_O),
        ('b_U', model.b_U)
    ]:
        network_stats['parameters'][name] = get_tensor_stats(param, name)
    
    # Get model predictions and collect activation statistics
    with th.no_grad():
        logits, activation_stats = forward_fn(model, all_sequences, collect_stats=True)
        network_stats['activations'] = activation_stats
        
        probs = F.softmax(logits, dim=-1)[:, -1]
        predictions = probs.argmax(-1)
        true_labels = all_sequences.max(-1).values
        
        correct_mask = (predictions == true_labels)
        accuracy = float(correct_mask.float().mean())
        soft_accuracy = float(probs[th.arange(len(all_sequences)), true_labels].mean())
        
        # Find error cases
        error_indices = th.where(~correct_mask)[0]
        error_cases = []
        
        for idx in error_indices:
            sequence = all_sequences[idx]
            pred = predictions[idx].item()
            true_label = true_labels[idx].item()
            confidence = probs[idx, pred].item()
            
            error_info = {
                'sequence': sequence.tolist(),
                'predicted': pred,
                'true_label': true_label,
                'confidence': confidence,
                'full_probs': probs[idx].tolist()
            }
            error_cases.append(error_info)
        
        # Print network statistics
        print("\n=== Network Statistics ===")
        print("\nParameter Statistics:")
        for name, stats in network_stats['parameters'].items():
            print(f"\n{name}:")
            for key, value in stats.items():
                if key != 'name':
                    print(f"  {key}: {value}")
        
        print("\nActivation Statistics (averaged over sequence):")
        for key in network_stats['activations']:
            print(f"\n{key}:")
            avg_stats = defaultdict(float)
            count = 0
            for stats in network_stats['activations'][key]:
                for stat_key, stat_value in stats.items():
                    if isinstance(stat_value, (int, float)):
                        avg_stats[stat_key] += stat_value
                        count += 1
            
            if count > 0:
                for stat_key, stat_sum in avg_stats.items():
                    if stat_key not in ['name', 'shape']:
                        print(f"  {stat_key}: {stat_sum / count:.6f}")
        
        print(f"\nModel Performance:")
        print(f"Total Accuracy: {accuracy:.4f}")
        print(f"Soft Accuracy: {soft_accuracy:.4f}")
        print(f"Number of errors: {len(error_cases)} out of {len(all_sequences)} sequences")
        
        return error_cases, accuracy, soft_accuracy, network_stats

def main():
    from utils_cleaned import get_model, generate_all_sequences
    
    th.set_grad_enabled(False)
    
    # Initialize model
    model = get_model('2-32')  # or '2-1500'
    
    # Generate all sequences
    all_sequences = generate_all_sequences(model.cfg.d_vocab, model.cfg.n_ctx)
    
    # Analyze model errors and collect network statistics
    error_cases, accuracy, soft_accuracy, network_stats = analyze_model_errors(
        model, forward, all_sequences
    )
    
    return error_cases, network_stats

if __name__ == "__main__":
    main()