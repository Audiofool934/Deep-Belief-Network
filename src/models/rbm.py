import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from typing import Optional, Dict, List


class RBM(nn.Module):
    def __init__(self, n_visible, n_hidden, device=None, training_method="cd"):
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.training_method = training_method  # 'cd' or 'mle'

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print(f"RBM using device: {self.device}")

        # Initialize weights and biases
        self.W = nn.Parameter(
            torch.randn(n_hidden, n_visible, device=self.device) * 0.1
        )
        self.h_bias = nn.Parameter(torch.zeros(n_hidden, device=self.device))
        self.v_bias = nn.Parameter(torch.zeros(n_visible, device=self.device))

        self.to(self.device)

    def sample_h(self, v):
        """Sample hidden units given visible units."""
        v = v.float().to(self.device)
        prob_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return prob_h, torch.bernoulli(prob_h)

    def sample_v(self, h):
        """Sample visible units given hidden units."""
        h = h.float().to(self.device)
        prob_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return prob_v, torch.bernoulli(prob_v)

    def forward(self, v):
        """Perform one Gibbs sampling step (v -> h -> v')."""
        # Ensure v is float
        v = v.float().to(self.device)
        prob_h, h_sample = self.sample_h(v)
        prob_v, v_sample = self.sample_v(h_sample)
        # Return the probability for smoother reconstruction error,
        # or v_sample for sampled reconstruction
        # Let's return prob_v for MSE calculation consistency
        return prob_v  # Or return v_sample if you prefer error on samples

    def free_energy(self, v):
        """Compute the free energy of a visible vector."""
        v = v.float().to(self.device)
        vbias_term = torch.matmul(v, self.v_bias)
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vbias_term - hidden_term

    def contrastive_divergence(self, v, k=1):
        """Perform k-step Contrastive Divergence."""
        v = v.float().to(self.device)
        v_k = v.clone()
        batch_size = v.size(0)

        # --- Gibbs Sampling ---
        # Store positive phase hidden probabilities/samples
        prob_h_pos, _ = self.sample_h(v)

        # Run k steps of Gibbs sampling
        for _ in range(k):
            _, h_k = self.sample_h(v_k)
            _, v_k = self.sample_v(h_k)

        v_k = v_k.detach()  # Detach to avoid gradient flow

        # Store negative phase hidden probabilities after k steps
        prob_h_neg, _ = self.sample_h(v_k)  # Get probabilities based on v_k

        # --- Gradient Calculation (using probabilities, applying sign fix) ---
        # Calculate positive and negative correlations <vh>
        # Using probabilities here, could also use samples (h_sample_pos, h_k)
        pos_phase = torch.matmul(prob_h_pos.t(), v)
        neg_phase = torch.matmul(prob_h_neg.t(), v_k)

        # Update gradients (corrected signs for SGD minimization)
        self.W.grad = (neg_phase - pos_phase) / batch_size
        self.v_bias.grad = torch.mean(v_k - v, dim=0)
        self.h_bias.grad = torch.mean(prob_h_neg - prob_h_pos, dim=0)

    def maximum_likelihood(self, v, mcmc_steps=100):
        """Perform maximum likelihood estimation using MCMC sampling."""
        v = v.float().to(self.device)
        batch_size = v.size(0)

        # Generate initial sample from model for MCMC chain
        # Use current v_bias probabilities for a slightly better start than random
        # Or keep random init: v_model = torch.bernoulli(torch.ones_like(v) * 0.5).to(self.device)

        with torch.no_grad():
            initial_prob_v = torch.sigmoid(self.v_bias.repeat(batch_size, 1))
            v_model = torch.bernoulli(initial_prob_v).to(self.device)

        # Run MCMC chain
        for _ in range(mcmc_steps):
            _, h_model = self.sample_h(v_model)
            _, v_model = self.sample_v(h_model)
        v_model = v_model.detach()  # Detach after chain finishes

        # Compute positive phase using data
        prob_h_pos, _ = self.sample_h(v)
        pos_phase = torch.matmul(prob_h_pos.t(), v)

        # Compute negative phase using MCMC samples
        prob_h_neg, _ = self.sample_h(v_model)
        neg_phase = torch.matmul(prob_h_neg.t(), v_model)

        # Update gradients (corrected signs for SGD minimization)
        self.W.grad = (neg_phase - pos_phase) / batch_size
        self.v_bias.grad = torch.mean(v_model - v, dim=0)
        self.h_bias.grad = torch.mean(prob_h_neg - prob_h_pos, dim=0)

    def train_rbm(
        self,
        data_loader,
        lr=0.01,
        k=1,
        mcmc_steps=100,
        epochs=10,
        record_metrics: bool = True,
    ) -> Optional[Dict[str, List[float]]]:
        """
        Train the RBM using selected method and optionally monitor metrics.

        Args:
            data_loader: DataLoader providing training batches.
            lr: Learning rate for the optimizer.
            k: Number of Gibbs steps for Contrastive Divergence.
            mcmc_steps: Number of MCMC steps for Maximum Likelihood training.
            epochs: Number of training epochs.
            record_metrics: If True, calculate and store average free energy
                             and reconstruction error per epoch. Set to False
                             to save computation if only final weights are needed.

        Returns:
            A dictionary containing the history of 'epoch', 'avg_free_energy',
            and 'recon_error' if record_metrics is True. Otherwise, returns None.
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=lr)
        training_method = self.training_method.lower()

        print(f"Training RBM using {training_method} method")
        training_loop = trange(epochs, desc="Initializing...")

        # Initialize history only if recording metrics
        history: Optional[Dict[str, List[float]]] = None
        if record_metrics:
            history = {"epoch": [], "avg_free_energy": [], "recon_error": []}

        for epoch in training_loop:
            # Initialize accumulators only if needed
            epoch_free_energy = 0.0
            epoch_recon_error = 0.0
            num_batches = 0

            for batch in data_loader:
                # Ensure data is flattened and on the correct device
                v = batch[0].view(batch[0].size(0), -1).to(self.device)
                v = v.float()  # Ensure input is float

                # --- RBM Parameter Update ---
                optimizer.zero_grad()

                if training_method == "cd":
                    self.contrastive_divergence(v, k)
                elif training_method == "mle":
                    self.maximum_likelihood(v, mcmc_steps)
                else:
                    raise ValueError(f"Unknown training method: {training_method}")

                optimizer.step()
                # --- End RBM Update ---

                # --- Optional Metric Calculation ---
                if record_metrics:
                    with torch.no_grad():  # Disable gradient calculation for monitoring
                        # 1. Average Free Energy
                        current_free_energy = torch.mean(self.free_energy(v))
                        epoch_free_energy += current_free_energy.item()

                        # 2. Reconstruction Error (Mean Squared Error)
                        v_reconstructed_prob = self.forward(v)
                        current_recon_error = F.mse_loss(
                            v_reconstructed_prob, v, reduction="mean"
                        )
                        epoch_recon_error += current_recon_error.item()
                # --- End Metric Calculation ---

                num_batches += 1

            # --- Epoch End Reporting ---
            if (
                record_metrics and history is not None
            ):  # Check history is not None for type safety
                avg_free_energy = (
                    epoch_free_energy / num_batches if num_batches > 0 else 0
                )
                avg_recon_error = (
                    epoch_recon_error / num_batches if num_batches > 0 else 0
                )

                history["epoch"].append(epoch + 1)
                history["avg_free_energy"].append(avg_free_energy)
                history["recon_error"].append(avg_recon_error)

                training_loop.set_description(
                    f"Epoch {epoch+1}/{epochs}, Avg FE: {avg_free_energy:.4f}, Recon Err: {avg_recon_error:.4f}"
                )
            else:
                # Provide basic progress update if not recording metrics
                training_loop.set_description(f"Epoch {epoch+1}/{epochs} Completed")
            # --- End Epoch Reporting ---

        print("Training finished.")
        return history  # Returns the dictionary or None
