import torch
import torch.nn.functional as F
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Dict, List, Optional, Tuple


def binarize(x, threshold=0.5):
    """
    Convert input tensor to binary (0 or 1) using the given threshold.

    Args:
        x: Input tensor
        threshold: Value above which the output will be 1, otherwise 0

    Returns:
        Binary tensor of same shape as input
    """
    return (x > threshold).float()


class BinarizedDataLoader:
    """
    Wrapper for PyTorch DataLoader that binarizes the data on-the-fly.
    """

    def __init__(self, dataloader, threshold=0.5):
        self.dataloader = dataloader
        self.threshold = threshold
        self.iterator = iter(dataloader)

    def __iter__(self):
        self.iterator = iter(self.dataloader)
        return self

    def __next__(self):
        batch = next(self.iterator)
        # Binarize and flatten the images
        n_visible = batch[0].shape[1] * batch[0].shape[2] * batch[0].shape[3]
        binary_batch = binarize(batch[0], self.threshold).view(-1, n_visible)
        return (binary_batch, batch[1])

    def __len__(self):
        return len(self.dataloader)


def visualize_training_history(history: Dict[str, List[float]]):
    """
    Visualize the training history of the RBM.

    Args:
        history: Dictionary containing 'epoch', 'avg_free_energy', and 'recon_error'
    """
    if (
        not history
        or "epoch" not in history
        or "avg_free_energy" not in history
        or "recon_error" not in history
    ):
        print(
            "No training history available. Make sure record_metrics=True when calling train_rbm()."
        )
        return

    # Create a figure with two subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))

    # Plot 1: Free Energy
    ax1.plot(history["epoch"], history["avg_free_energy"], "b-", marker="o")
    ax1.set_xlabel("Epoch")
    ax1.set_ylabel("Average Free Energy")
    ax1.set_title("Free Energy vs. Epoch")
    ax1.grid(True, linestyle="--", alpha=0.7)

    # Plot 2: Reconstruction Error
    ax2.plot(history["epoch"], history["recon_error"], "r-", marker="o")
    ax2.set_xlabel("Epoch")
    ax2.set_ylabel("Reconstruction Error (MSE)")
    ax2.set_title("Reconstruction Error vs. Epoch")
    ax2.grid(True, linestyle="--", alpha=0.7)

    # Add a summary table with numeric values
    print("Training History Summary:")
    print(f"{'Epoch':<10}{'Free Energy':<20}{'Reconstruction Error':<20}")
    print("-" * 50)
    for i, epoch in enumerate(history["epoch"]):
        print(
            f"{epoch:<10}{history['avg_free_energy'][i]:<20.4f}{history['recon_error'][i]:<20.4f}"
        )

    plt.tight_layout()
    plt.show()

    # Calculate improvements
    if len(history["epoch"]) > 1:
        fe_improvement = (
            (
                (history["avg_free_energy"][0] - history["avg_free_energy"][-1])
                / abs(history["avg_free_energy"][0])
            )
            * 100
            if history["avg_free_energy"][0] != 0
            else 0
        )
        re_improvement = (
            (
                (history["recon_error"][0] - history["recon_error"][-1])
                / history["recon_error"][0]
            )
            * 100
            if history["recon_error"][0] != 0
            else 0
        )

        print(f"\nOverall Improvements:")
        print(f"Free Energy: {fe_improvement:.2f}%")
        print(f"Reconstruction Error: {re_improvement:.2f}%")


def visualize_weights(
    rbm, num_to_plot=100, cmap="viridis", title="RBM Learned Features", figsize=(12, 12)
):
    """
    Visualize the weights (learned features) of the RBM.

    Args:
        rbm: Trained RBM model
        num_to_plot: Number of hidden units to visualize
        cmap: Matplotlib colormap to use for visualization
        title: Title for the plot
        figsize: Figure size as (width, height) tuple

    Returns:
        matplotlib figure and axes for further customization
    """
    # Visualize the learned features (weights)
    fig = plt.figure(figsize=figsize)
    W = rbm.W.detach().cpu().numpy()

    # Limit to the specified number of hidden units
    num_to_plot = min(num_to_plot, rbm.n_hidden)
    side = int(np.ceil(np.sqrt(num_to_plot)))

    axes = []
    for i in range(num_to_plot):
        ax = fig.add_subplot(side, side, i + 1)
        # Reshape the weight vector for hidden unit i into a 28x28 image
        feature = W[i].reshape(28, 28)

        # Calculate weight statistics for this feature
        w_min, w_max = feature.min(), feature.max()
        w_mean, w_std = feature.mean(), feature.std()

        # Plot the feature
        img = ax.imshow(feature, cmap=cmap)
        ax.set_title(f"Unit {i}", fontsize=8)
        ax.axis("off")
        axes.append(ax)

    plt.suptitle(title, fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)

    # Add a colorbar to the figure
    fig.colorbar(img, ax=axes, orientation="horizontal", fraction=0.02, pad=0.05)

    plt.show()
    return fig, axes


def generate_samples(rbm, num_samples=10, gibbs_steps=1000):
    """
    Generate new samples from the trained RBM using Gibbs sampling.

    Args:
        rbm: Trained RBM model
        num_samples: Number of samples to generate
        gibbs_steps: Number of Gibbs sampling steps

    Returns:
        Generated samples
    """
    # Start with random noise
    v = torch.rand(num_samples, rbm.n_visible).to(rbm.device)
    v = (v > 0.5).float()  # Binarize

    # Perform Gibbs sampling
    for _ in trange(gibbs_steps):
        _, h = rbm.sample_h(v)
        _, v = rbm.sample_v(h)

    return v


def visualize_reconstructions(rbm, test_loader, device, n_samples=10):
    """
    Visualize original test images and their reconstructions.

    Args:
        rbm: Trained RBM model
        test_loader: DataLoader for test dataset
        device: Device to use for computations
        n_samples: Number of samples to visualize
    """
    # Get some test images
    test_batch = next(iter(test_loader))
    test_images = test_batch[0][:n_samples].view(-1, rbm.n_visible).to(device)
    test_labels = test_batch[1][:n_samples]

    # Binarize the images
    binary_test_images = binarize(test_images)

    # Reconstruct images
    with torch.no_grad():
        # Get hidden activation probabilities
        h_prob, h_sample = rbm.sample_h(binary_test_images)
        # Reconstruct from hidden activations
        v_prob, v_sample = rbm.sample_v(h_sample)

    # Visualize original and reconstructed images
    plt.figure(figsize=(12, 6))
    for i in range(n_samples):
        # Original image
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(binary_test_images[i].cpu().view(28, 28), cmap="gray")
        plt.title(f"Original: {test_labels[i]}")
        plt.axis("off")

        # Reconstructed image
        plt.subplot(2, n_samples, i + n_samples + 1)
        plt.imshow(v_prob[i].cpu().view(28, 28), cmap="gray")
        plt.title("Recon")
        plt.axis("off")

    plt.tight_layout()
    plt.show()


def analyze_digit_features(rbm, test_dataset, device, n_visible=784):
    """
    Analyze and visualize features for each digit class.

    Args:
        rbm: Trained RBM model
        test_dataset: Test dataset
        device: Device to use for computations
        n_visible: Number of visible units

    Returns:
        Tuple of (digit_samples, digit_features)
    """
    # Get test samples for each digit class (0-9)
    digit_samples = {}
    digit_features = {}

    # Find representative samples for each digit
    for digit in range(10):
        # Find indices of all examples of this digit in the test set
        indices = [i for i, (_, label) in enumerate(test_dataset) if label == digit]
        if indices:
            # Take the first few examples
            samples = [test_dataset[i][0].view(-1, n_visible) for i in indices[:5]]
            samples = torch.cat(samples).to(device)

            # Binarize
            binary_samples = binarize(samples)

            # Get hidden activations
            with torch.no_grad():
                h_activations, _ = rbm.sample_h(binary_samples)

            # Store
            digit_samples[digit] = binary_samples
            digit_features[digit] = h_activations.mean(dim=0).cpu().numpy()

    return digit_samples, digit_features


def visualize_digit_features(rbm, digit_samples, digit_features, n_top_features=5):
    """
    Visualize top features for each digit.

    Args:
        rbm: Trained RBM model
        digit_samples: Dictionary mapping digit to samples
        digit_features: Dictionary mapping digit to feature activations
        n_top_features: Number of top features to show per digit
    """
    # Plot the top features for each digit
    plt.figure(figsize=(15, 20))
    W = rbm.W.detach().cpu().numpy()

    for digit in range(10):
        # Get the activation strengths for this digit
        if digit in digit_features:
            activations = digit_features[digit]

            # Find indices of top activated features
            top_indices = np.argsort(activations)[-n_top_features:]

            # Show example of the digit
            plt.subplot(10, n_top_features + 1, digit * (n_top_features + 1) + 1)
            if digit in digit_samples:
                plt.imshow(digit_samples[digit][0].cpu().view(28, 28), cmap="gray")
                plt.title(f"Digit {digit}")
                plt.axis("off")

            # Show top features for this digit
            for i, idx in enumerate(top_indices):
                plt.subplot(
                    10, n_top_features + 1, digit * (n_top_features + 1) + i + 2
                )
                feature = W[idx].reshape(28, 28)
                plt.imshow(feature, cmap="viridis")
                plt.title(f"Activation: {activations[idx]:.2f}")
                plt.axis("off")

    plt.suptitle("Top Features for Each Digit", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.95)
    plt.show()


def visualize_feature_heatmap(digit_features, n_hidden=256):
    """
    Create a feature activation heatmap for all digits.

    Args:
        digit_features: Dictionary mapping digit to feature activations
        n_hidden: Number of hidden units
    """
    plt.figure(figsize=(12, 8))
    activation_matrix = np.zeros((10, min(50, n_hidden)))

    # Get top 50 most variable features across digits
    feature_variances = []
    for i in range(n_hidden):
        feature_vals = [digit_features[d][i] for d in range(10) if d in digit_features]
        if feature_vals:
            feature_variances.append((i, np.var(feature_vals)))

    # Sort by variance
    feature_variances.sort(key=lambda x: x[1], reverse=True)
    top_feature_indices = [idx for idx, _ in feature_variances[:50]]

    # Fill activation matrix
    for i, digit in enumerate(range(10)):
        if digit in digit_features:
            for j, feature_idx in enumerate(top_feature_indices):
                activation_matrix[i, j] = digit_features[digit][feature_idx]

    # Plot heatmap
    im = plt.imshow(activation_matrix, cmap="viridis", aspect="auto")
    plt.colorbar(im, label="Activation Strength")
    plt.xlabel("Feature Index")
    plt.ylabel("Digit")
    plt.title("Feature Activation by Digit")
    plt.yticks(np.arange(10))
    plt.tight_layout()
    plt.show()


def visualize_generated_samples(generated_samples, n_samples=20):
    """
    Visualize samples generated from the RBM.

    Args:
        generated_samples: Tensor of generated samples
        n_samples: Number of samples to visualize
    """
    plt.figure(figsize=(15, 6))
    for i in range(min(n_samples, len(generated_samples))):
        plt.subplot(2, n_samples // 2, i + 1)
        plt.imshow(generated_samples[i].detach().cpu().view(28, 28), cmap="gray")
        plt.axis("off")

    plt.suptitle("Generated Samples", fontsize=16)
    plt.tight_layout()
    plt.subplots_adjust(top=0.9)
    plt.show()
