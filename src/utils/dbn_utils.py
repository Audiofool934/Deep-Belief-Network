import torch
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Dict, List, Tuple, Optional

def train_layer_wise(dbn, data_loader, lr=0.01, k=1, epochs_per_layer=10):
    """
    Train a Deep Belief Network using greedy layer-wise training.
    
    Args:
        dbn: Deep Belief Network model
        data_loader: DataLoader containing the training data
        lr: Learning rate for each RBM
        k: Number of Gibbs sampling steps
        epochs_per_layer: Number of epochs to train each layer
        
    Returns:
        Dictionary containing training history for each layer
    """
    layer_histories = {}
    
    # Start with the raw input data
    current_data_loader = data_loader
    
    # Train each layer in turn
    for i, rbm in enumerate(dbn.rbm_layers):
        print(f"\nTraining layer {i+1}/{len(dbn.rbm_layers)} (RBM with {rbm.n_visible} visible and {rbm.n_hidden} hidden units)")
        
        # Train current RBM
        history = rbm.train_rbm(
            data_loader=current_data_loader,
            lr=lr,
            k=k,
            epochs=epochs_per_layer,
            record_metrics=True
        )
        
        layer_histories[i] = history
        
        # Transform data for next layer (if not the last layer)
        if i < len(dbn.rbm_layers) - 1:
            print(f"Transforming data for layer {i+2}...")
            current_data_loader = transform_data_for_next_layer(rbm, current_data_loader)
    
    return layer_histories


def transform_data_for_next_layer(rbm, data_loader):
    """
    Transform dataset through current RBM to create input for the next layer.
    
    Args:
        rbm: Trained RBM model
        data_loader: DataLoader with current input data
        
    Returns:
        New DataLoader with transformed data
    """
    transformed_data = []
    labels = []
    
    with torch.no_grad():
        for batch, batch_labels in data_loader:
            # Get hidden activation probabilities
            h_prob, _ = rbm.sample_h(batch)
            transformed_data.append(h_prob)
            labels.append(batch_labels)
    
    # Convert to tensors
    transformed_data = torch.cat(transformed_data)
    labels = torch.cat(labels)
    
    # Create a simple dataset from tensors
    transformed_dataset = torch.utils.data.TensorDataset(transformed_data, labels)
    
    # Create dataloader with the same batch size
    new_data_loader = torch.utils.data.DataLoader(
        transformed_dataset,
        batch_size=data_loader.batch_size if hasattr(data_loader, 'batch_size') else 64,
        shuffle=True
    )
    
    return new_data_loader


def visualize_layer_training_history(layer_histories, figsize=(15, 10)):
    """
    Visualize training history across all layers of the DBN.
    
    Args:
        layer_histories: Dictionary containing training history for each layer
        figsize: Figure size tuple (width, height)
    """
    n_layers = len(layer_histories)
    
    # Create figure with 2 rows (free energy and reconstruction error)
    fig, axes = plt.subplots(2, n_layers, figsize=figsize)
    
    # If only one layer, make sure axes is 2D
    if n_layers == 1:
        axes = axes.reshape(2, 1)
    
    for layer_idx, history in layer_histories.items():
        # Plot free energy for this layer
        axes[0, layer_idx].plot(history['epoch'], history['avg_free_energy'], 'b-', marker='o')
        axes[0, layer_idx].set_title(f'Layer {layer_idx+1}: Free Energy')
        axes[0, layer_idx].set_xlabel('Epoch')
        axes[0, layer_idx].set_ylabel('Free Energy')
        axes[0, layer_idx].grid(True, linestyle='--', alpha=0.7)
        
        # Plot reconstruction error for this layer
        axes[1, layer_idx].plot(history['epoch'], history['recon_error'], 'r-', marker='o')
        axes[1, layer_idx].set_title(f'Layer {layer_idx+1}: Reconstruction Error')
        axes[1, layer_idx].set_xlabel('Epoch')
        axes[1, layer_idx].set_ylabel('Error')
        axes[1, layer_idx].grid(True, linestyle='--', alpha=0.7)
    
    plt.tight_layout()
    plt.show()


def visualize_dbn_activations(dbn, input_data, layer_idx=None, cmap='viridis'):
    """
    Visualize activations of units in different layers of the DBN.
    
    Args:
        dbn: Trained DBN model
        input_data: Batch of input data
        layer_idx: Specific layer to visualize (None for all layers)
        cmap: Matplotlib colormap to use
    """
    # Forward pass through the DBN
    activations = []
    curr_data = input_data
    
    with torch.no_grad():
        for i, rbm in enumerate(dbn.rbm_layers):
            h_prob, _ = rbm.sample_h(curr_data)
            activations.append(h_prob)
            curr_data = h_prob
    
    # Determine which layers to visualize
    layers_to_show = [layer_idx] if layer_idx is not None else range(len(dbn.rbm_layers))
    
    for i in layers_to_show:
        if i >= len(activations):
            print(f"Layer {i+1} doesn't exist in the model.")
            continue
            
        plt.figure(figsize=(10, 6))
        
        # Get activations for the current layer
        layer_acts = activations[i].cpu().numpy()
        
        # Display activations for up to 10 examples
        n_examples = min(10, layer_acts.shape[0])
        for j in range(n_examples):
            plt.subplot(2, 5, j+1)
            
            # Reshape if small enough to form a square image
            n_units = layer_acts.shape[1]
            side_len = int(np.sqrt(n_units))
            
            if side_len**2 == n_units:  # Perfect square
                plt.imshow(layer_acts[j].reshape(side_len, side_len), cmap=cmap)
            else:
                # Plot as a 1D array
                plt.imshow(layer_acts[j].reshape(1, -1), cmap=cmap, aspect='auto')
                
            plt.title(f"Example {j+1}")
            plt.axis('off')
        
        plt.suptitle(f"Layer {i+1} Activations ({layer_acts.shape[1]} units)", fontsize=16)
        plt.tight_layout()
        plt.subplots_adjust(top=0.9)
        plt.show()
        
        # Show activation statistics
        plt.figure(figsize=(10, 4))
        
        # Mean activation across examples
        plt.subplot(1, 2, 1)
        mean_act = layer_acts.mean(axis=0)
        plt.bar(range(min(50, len(mean_act))), mean_act[:50])
        plt.xlabel('Unit Index')
        plt.ylabel('Mean Activation')
        plt.title(f'Layer {i+1}: Mean Unit Activation (first 50 units)')
        
        # Activation distribution
        plt.subplot(1, 2, 2)
        plt.hist(layer_acts.flatten(), bins=50)
        plt.xlabel('Activation Value')
        plt.ylabel('Frequency')
        plt.title(f'Layer {i+1}: Activation Distribution')
        
        plt.tight_layout()
        plt.show()


def visualize_generative_weights(dbn, layer_idx=0, num_to_plot=16, figsize=(10, 10)):
    """
    Visualize the generative weights of a specific layer in the DBN.
    
    Args:
        dbn: Trained DBN model
        layer_idx: Layer index to visualize
        num_to_plot: Number of hidden units to visualize
        figsize: Figure size tuple (width, height)
    """
    if layer_idx >= len(dbn.rbm_layers):
        print(f"Layer {layer_idx+1} doesn't exist in the model.")
        return
    
    rbm = dbn.rbm_layers[layer_idx]
    W = rbm.W.detach().cpu().numpy()
    
    # Limit the number of units to visualize
    num_to_plot = min(num_to_plot, rbm.n_hidden)
    
    # Determine grid size
    grid_size = int(np.ceil(np.sqrt(num_to_plot)))
    
    plt.figure(figsize=figsize)
    for i in range(num_to_plot):
        plt.subplot(grid_size, grid_size, i + 1)
        
        # For the first layer, we can reshape to the input dimensions
        if layer_idx == 0 and np.sqrt(rbm.n_visible).is_integer():
            img_side = int(np.sqrt(rbm.n_visible))
            plt.imshow(W[i].reshape(img_side, img_side), cmap='viridis')
        else:
            # For higher layers, just show the weight pattern
            plt.imshow(W[i].reshape(1, -1), aspect='auto', cmap='viridis')
        
        plt.axis('off')
        plt.title(f"Unit {i}")
    
    plt.suptitle(f"Layer {layer_idx+1} Generative Weights", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()


def generate_samples_from_dbn(dbn, num_samples=10, gibbs_steps=2000):
    """
    Generate samples from the DBN using its top-level RBM and 
    propagating the samples back to the visible layer.
    
    Args:
        dbn: Trained DBN model
        num_samples: Number of samples to generate
        gibbs_steps: Number of Gibbs sampling steps for the top RBM
        
    Returns:
        Generated samples after propagating through all layers
    """
    # Start with the top-level RBM
    top_rbm = dbn.rbm_layers[-1]
    
    # 1. First generate samples from the top RBM
    print("Generating samples from top-level RBM...")
    h_samples = torch.rand(num_samples, top_rbm.n_hidden).to(top_rbm.device)
    h_samples = (h_samples > 0.5).float()
    
    # Perform Gibbs sampling in the top-level RBM
    v_samples = None
    for _ in trange(gibbs_steps):
        v_samples, _ = top_rbm.sample_v(h_samples)
        h_samples, _ = top_rbm.sample_h(v_samples)
    
    # 2. Propagate the samples backwards through the network
    print("Propagating samples through the network...")
    for i in range(len(dbn.rbm_layers) - 2, -1, -1):
        rbm = dbn.rbm_layers[i]
        v_samples, _ = rbm.sample_v(v_samples)
    
    return v_samples