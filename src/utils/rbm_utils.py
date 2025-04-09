import torch
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
from tqdm import trange
from typing import Dict, List
import math


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


def MNIST():
    # Define transformation
    transform = transforms.Compose(
        [
            transforms.ToTensor(),
        ]
    )

    # Load MNIST dataset
    train_dataset = datasets.MNIST(
        root="../data", train=True, download=True, transform=transform
    )

    test_dataset = datasets.MNIST(
        root="../data", train=False, download=True, transform=transform
    )

    return train_dataset, test_dataset


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
        or "recon_error_features" not in history
    ):
        print(
            "No training history available. Make sure record_metrics=True when calling train_rbm()."
        )
        return

    num_subplots = 2
    if "recon_error_labels" in history:
        num_subplots += 1
    if "classification_accuracy" in history and not all(
        torch.isnan(torch.tensor(history["classification_accuracy"]))
    ):
        num_subplots += 1

    fig, axes = plt.subplots(1, num_subplots, figsize=(5 * num_subplots, 5))

    ax_index = 0

    axes[ax_index].plot(history["epoch"], history["avg_free_energy"], "b-", marker="o")
    axes[ax_index].set_xlabel("Epoch")
    axes[ax_index].set_ylabel("Average Free Energy")
    axes[ax_index].set_title("Free Energy vs. Epoch")
    axes[ax_index].grid(True, linestyle="--", alpha=0.7)
    ax_index += 1

    axes[ax_index].plot(
        history["epoch"], history["recon_error_features"], "r-", marker="o"
    )
    axes[ax_index].set_xlabel("Epoch")
    axes[ax_index].set_ylabel("Reconstruction Error (MSE) - Features")
    axes[ax_index].set_title("Feature Reconstruction Error vs. Epoch")
    axes[ax_index].grid(True, linestyle="--", alpha=0.7)
    ax_index += 1

    if "recon_error_labels" in history:
        axes[ax_index].plot(
            history["epoch"], history["recon_error_labels"], "g-", marker="o"
        )
        axes[ax_index].set_xlabel("Epoch")
        axes[ax_index].set_ylabel("Reconstruction Error (MSE) - Labels")
        axes[ax_index].set_title("Label Reconstruction Error vs. Epoch")
        axes[ax_index].grid(True, linestyle="--", alpha=0.7)
        ax_index += 1

    if "classification_accuracy" in history and not all(
        torch.isnan(torch.tensor(history["classification_accuracy"]))
    ):
        valid_accuracy = [
            acc
            for acc in history["classification_accuracy"]
            if not torch.isnan(torch.tensor(acc))
        ]
        valid_epochs = [
            epoch
            for epoch, acc in zip(history["epoch"], history["classification_accuracy"])
            if not torch.isnan(torch.tensor(acc))
        ]
        axes[ax_index].plot(valid_epochs, valid_accuracy, "m-", marker="o")
        axes[ax_index].set_xlabel("Epoch")
        axes[ax_index].set_ylabel("Classification Accuracy (%)")
        axes[ax_index].set_title("Classification Accuracy vs. Epoch")
        axes[ax_index].grid(True, linestyle="--", alpha=0.7)

    print("Training History Summary:")
    header = f"{'Epoch':<10}{'Free Energy':<20}{'Recon Err (Feat)':<20}"
    if "recon_error_labels" in history:
        header += f"{'Recon Err (Lbl)':<20}"
    if "classification_accuracy" in history:
        header += f"{'Classification Acc':<20}"
    print(header)
    print(
        "-"
        * (
            10
            + 20
            * (
                2
                + ("recon_error_labels" in history)
                + ("classification_accuracy" in history)
            )
        )
    )
    for i, epoch in enumerate(history["epoch"]):
        if i % 5 == 0:
            row = f"{epoch:<10}{history['avg_free_energy'][i]:<20.4f}{history['recon_error_features'][i]:<20.4f}"
            if "recon_error_labels" in history:
                row += f"{history['recon_error_labels'][i]:<20.4f}"
            if "classification_accuracy" in history:
                if torch.isnan(torch.tensor(history["classification_accuracy"][i])):
                    row += f"{'N/A':<20}"
                else:
                    row += f"{history['classification_accuracy'][i]:<20.2f}"
            print(row)

    plt.tight_layout()
    plt.show()

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
        re_improvement_features = (
            (
                (
                    history["recon_error_features"][0]
                    - history["recon_error_features"][-1]
                )
                / history["recon_error_features"][0]
            )
            * 100
            if history["recon_error_features"][0] != 0
            else 0
        )

        print(f"\nOverall Improvements:")
        print(f"Free Energy: {fe_improvement:.2f}%")
        print(f"Reconstruction Error (Features): {re_improvement_features:.2f}%")

        if "recon_error_labels" in history:
            re_improvement_labels = (
                (
                    (
                        history["recon_error_labels"][0]
                        - history["recon_error_labels"][-1]
                    )
                    / history["recon_error_labels"][0]
                )
                * 100
                if history["recon_error_labels"][0] != 0
                else 0
            )
            print(f"Reconstruction Error (Labels): {re_improvement_labels:.2f}%")

        if "classification_accuracy" in history and not all(
            torch.isnan(torch.tensor(history["classification_accuracy"]))
        ):
            valid_accuracy = [
                acc
                for acc in history["classification_accuracy"]
                if not torch.isnan(torch.tensor(acc))
            ]
            acc_improvement = valid_accuracy[-1] - valid_accuracy[0]
            print(f"Classification Accuracy: {acc_improvement:.2f}%")


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
    features, labels = next(iter(test_loader))
    v_input = rbm.prepare_input_data(features, labels)

    # Reconstruct images
    with torch.no_grad():
        v_prob = rbm.forward(v_input)
        v_recon_features = v_prob[:, : rbm.n_visible_features]

    # Visualize original and reconstructed images
    plt.figure(figsize=(12, 6))
    for i in range(n_samples):
        # Original image
        plt.subplot(2, n_samples, i + 1)
        plt.imshow(features[i].cpu().view(28, 28), cmap="gray")
        plt.title(f"Original: {labels[i]}")
        plt.axis("off")

        # Reconstructed image
        plt.subplot(2, n_samples, i + n_samples + 1)
        plt.imshow(v_recon_features[i].cpu().view(28, 28), cmap="gray")
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
