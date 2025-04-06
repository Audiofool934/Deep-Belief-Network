import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from typing import Optional, Dict, List, Tuple


class RBM(nn.Module):
    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super(RBM, self).__init__()
        self.n_visible = n_visible
        self.n_hidden = n_hidden

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

        self.W = nn.Parameter(
            torch.randn(n_hidden, n_visible, device=self.device) * 0.1
        )
        self.h_bias = nn.Parameter(torch.zeros(n_hidden, device=self.device))
        self.v_bias = nn.Parameter(torch.zeros(n_visible, device=self.device))

        self.to(self.device)

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample hidden units given visible units."""
        v = v.float().to(self.device)
        prob_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return prob_h, torch.bernoulli(prob_h)

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Sample visible units given hidden units."""
        h = h.float().to(self.device)
        prob_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return prob_v, torch.bernoulli(prob_v)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """Perform one Gibbs sampling step (v -> h -> v')."""
        v = v.float().to(self.device)
        prob_h, h_sample = self.sample_h(v)
        prob_v, v_sample = self.sample_v(h_sample)
        return prob_v

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """Compute the free energy of a visible vector."""
        v = v.float().to(self.device)
        vbias_term = torch.matmul(v, self.v_bias)  # Dot product for binary
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vbias_term - hidden_term

    def contrastive_divergence(self, v: torch.Tensor, k: int = 1) -> None:
        """
        Perform k-step Contrastive Divergence using Free Energy difference.
        Gradients are computed automatically via autograd.
        """
        v = v.float().to(self.device)
        v_0 = v

        v_k = v.clone()
        for _ in range(k):
            _, h_k = self.sample_h(v_k)
            _, v_k = self.sample_v(h_k)
        v_k = v_k.detach()

        fe_positive = self.free_energy(v_0)
        fe_negative = self.free_energy(v_k)

        loss = torch.mean(fe_positive - fe_negative)
        loss.backward()

    def maximum_likelihood(self, v: torch.Tensor, mcmc_steps: int = 100) -> None:
        """
        Perform maximum likelihood estimation using MCMC sampling and Free Energy difference.
        Gradients are computed automatically via autograd.
        """
        v = v.float().to(self.device)
        v_data = v
        batch_size = v.size(0)

        with torch.no_grad():
            if isinstance(self, GaussianBernoulliRBM):
                std_dev = self.get_std().detach()
                v_model = torch.normal(
                    mean=self.v_bias.repeat(batch_size, 1),
                    std=std_dev.repeat(batch_size, 1),
                ).to(self.device)
            else:
                initial_prob_v = torch.sigmoid(self.v_bias.repeat(batch_size, 1))
                v_model = torch.bernoulli(initial_prob_v).to(self.device)

            for _ in range(mcmc_steps):
                _, h_model = self.sample_h(v_model)
                _, v_model = self.sample_v(h_model)

        v_model = v_model.detach()

        fe_positive = self.free_energy(v_data)
        fe_negative = self.free_energy(v_model)

        loss = torch.mean(fe_positive - fe_negative)
        loss.backward()

    def train_rbm(
        self,
        data_loader: torch.utils.data.DataLoader,
        training_method: str = "cd",
        lr: float = 0.01,
        k: int = 1,
        mcmc_steps: int = 100,
        epochs: int = 10,
        record_metrics: bool = True,
        weight_decay: float = 0.0,
    ) -> Optional[Dict[str, List[float]]]:
        """
        Train the RBM using selected method, autograd, and optionally monitor metrics.

        Args:
            data_loader: DataLoader providing training data
            lr: Learning rate
            k: Number of CD steps (if using CD)
            mcmc_steps: Number of MCMC steps (if using MLE)
            epochs: Number of training epochs
            record_metrics: Whether to record training metrics
            weight_decay: L2 regularization parameter

        Returns:
            Optional dictionary containing training metrics if record_metrics=True
        """
        optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        training_method = training_method.lower()

        print(
            f"Training {self.__class__.__name__} using {training_method} method with Autograd"
        )
        training_loop = trange(epochs, desc="Initializing...")

        history: Optional[Dict[str, List[float]]] = None
        if record_metrics:
            history = {
                "epoch": [],
                "avg_free_energy": [],
                "recon_error": [],
                "avg_std_dev": [],
            }
            if isinstance(self, GaussianBernoulliRBM):
                history["avg_std_dev"] = []

        self.eps = 1e-8

        for epoch in training_loop:
            epoch_free_energy = 0.0
            epoch_recon_error = 0.0
            epoch_std_dev_sum = 0.0
            num_batches = 0
            items_processed = 0

            for batch in data_loader:
                v = batch[0].view(batch[0].size(0), -1).to(self.device)
                v = v.float()
                current_batch_size = v.size(0)

                optimizer.zero_grad()

                if training_method == "cd":
                    self.contrastive_divergence(v, k)
                elif training_method == "mle":
                    self.maximum_likelihood(v, mcmc_steps)
                else:
                    raise ValueError(f"Unknown training method: {training_method}")

                valid_gradients = True
                for p in self.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            print(
                                f"\nWarning: NaN or Inf gradient detected in param {p.shape} epoch {epoch+1}. Skipping update."
                            )
                            valid_gradients = False
                            break
                if not valid_gradients:
                    optimizer.zero_grad()
                    continue

                optimizer.step()

                if record_metrics:
                    with torch.no_grad():
                        current_free_energy = torch.mean(self.free_energy(v)).item()
                        epoch_free_energy += current_free_energy * current_batch_size

                        v_reconstructed_mean = self.forward(v)
                        current_recon_error = F.mse_loss(
                            v_reconstructed_mean, v, reduction="mean"
                        ).item()
                        epoch_recon_error += current_recon_error * current_batch_size

                        if isinstance(self, GaussianBernoulliRBM):
                            avg_batch_std = torch.mean(self.get_std()).item()
                            epoch_std_dev_sum += avg_batch_std * current_batch_size

                items_processed += current_batch_size
                num_batches += 1

            if record_metrics and history is not None and items_processed > 0:
                avg_free_energy = epoch_free_energy / items_processed
                avg_recon_error = epoch_recon_error / items_processed

                history["epoch"].append(epoch + 1)
                history["avg_free_energy"].append(avg_free_energy)
                history["recon_error"].append(avg_recon_error)

                desc = f"Epoch {epoch+1}/{epochs}, Avg FE: {avg_free_energy:.4f}, Recon Err: {avg_recon_error:.4f}"
                if isinstance(self, GaussianBernoulliRBM):
                    avg_std_dev = epoch_std_dev_sum / items_processed
                    history["avg_std_dev"].append(avg_std_dev)
                    desc += f", Avg Std: {avg_std_dev:.4f}"

                training_loop.set_description(desc)

            elif items_processed > 0:
                training_loop.set_description(f"Epoch {epoch+1}/{epochs} Completed")
            else:
                training_loop.set_description(
                    f"Epoch {epoch+1}/{epochs} - No items processed"
                )

        print("Training finished.")
        return history

class GaussianBernoulliRBM(RBM):
    """
    Gaussian-Bernoulli Restricted Boltzmann Machine (GRBM)
    with learnable standard deviation for each visible unit.

    Visible units are continuous (Gaussian), hidden units are binary (Bernoulli).
    Data normalization (e.g., zero mean, unit variance) is still recommended
    as a starting point.
    """

    def __init__(
        self,
        n_visible: int,
        n_hidden: int,
        device: Optional[torch.device] = None,
    ) -> None:
        super(GaussianBernoulliRBM, self).__init__(
            n_visible, n_hidden, device=device
        )
        self.log_std = nn.Parameter(torch.zeros(n_visible, device=self.device))
        print("Instantiated GaussianBernoulliRBM with learnable std dev.")
        self.eps = 1e-8

    def get_std(self) -> torch.Tensor:
        """Returns the standard deviation (exp(log_std))."""
        return torch.exp(self.log_std)

    def get_var(self) -> torch.Tensor:
        """Returns the variance (std^2). Clamped for stability."""
        return torch.exp(2 * self.log_std) + self.eps

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given continuous visible units.
        p(h_j=1 | v) = sigmoid( c_j + sum_i (W_ji * v_i / var_i) )

        Args:
            v: Visible unit values of shape [batch_size, n_visible]

        Returns:
            Tuple of (hidden unit probabilities, hidden unit samples)
        """
        v = v.float().to(self.device)
        var = self.get_var()

        v_scaled = v / var

        activation = F.linear(v_scaled, self.W, self.h_bias)
        prob_h = torch.sigmoid(activation)
        return prob_h, torch.bernoulli(prob_h)

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units given binary hidden units.
        v ~ N(mean = b + W.T @ h, variance = exp(2*log_std))

        Args:
            h: Hidden unit values of shape [batch_size, n_hidden]

        Returns:
            Tuple of (visible unit means, visible unit samples)
        """
        h = h.float().to(self.device)
        mean_v = F.linear(h, self.W.t(), self.v_bias)

        std_dev = self.get_std()
        std_dev_batch = std_dev.expand_as(mean_v)

        noise = torch.randn_like(mean_v, device=self.device)
        v_sample = mean_v + noise * std_dev_batch

        return mean_v, v_sample

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Perform one Gibbs sampling step (v -> h -> v'). Returns mean reconstruction.

        Args:
            v: Input visible units

        Returns:
            Mean reconstruction of visible units
        """
        v = v.float().to(self.device)
        prob_h, h_sample = self.sample_h(v)
        v_mean_recon, v_sample_recon = self.sample_v(h_sample)
        return v_mean_recon

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of a continuous visible vector.
        F(v) = sum_i (v_i - b_i)^2 / (2*var_i) - sum_j softplus( c_j + sum_i W_ji*v_i/var_i )

        Args:
            v: Visible unit values

        Returns:
            Free energy vector (one value per sample in the batch)
        """
        v = v.float().to(self.device)
        var = self.get_var()

        visible_term = torch.sum(0.5 * ((v - self.v_bias) ** 2) / var, dim=1)

        v_scaled = v / var
        wx_b = F.linear(v_scaled, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)

        return visible_term - hidden_term


# --- Example Usage (Optional - Updated) ---
if __name__ == "__main__":
    import torch.utils.data as data
    from sklearn.preprocessing import StandardScaler

    print("\n--- Testing GaussianBernoulliRBM with Learnable Std Dev ---")
    n_samples = 1000
    n_vis = 20
    n_hid = 10
    means = torch.randn(n_vis) * 3
    stds = torch.rand(n_vis) * 2 + 0.5
    dummy_data = torch.randn(n_samples, n_vis) * stds + means
    print(f"Target std devs (first 5): {stds[:5].numpy()}")

    scaler = StandardScaler()
    dummy_data_normalized = scaler.fit_transform(dummy_data.numpy())
    dummy_data_tensor = torch.tensor(dummy_data_normalized, dtype=torch.float32)
    print(f"Data mean after scaling (should be near 0): {dummy_data_tensor.mean():.4f}")
    print(f"Data std after scaling (should be near 1): {dummy_data_tensor.std():.4f}")

    dummy_labels = torch.zeros(n_samples)
    dataset = data.TensorDataset(dummy_data_tensor, dummy_labels)
    dataloader = data.DataLoader(dataset, batch_size=64, shuffle=True)

    grbm = GaussianBernoulliRBM(n_visible=n_vis, n_hidden=n_hid, training_method="cd")

    print(f"Model parameters on device: {next(grbm.parameters()).device}")

    history = grbm.train_rbm(dataloader, lr=0.005, k=1, epochs=50, record_metrics=True)

    if history:
        print("\nTraining History (Sample):")
        idx = min(5, len(history["epoch"]))
        for i in range(idx):
            print(
                f"Epoch {history['epoch'][i]}: FE={history['avg_free_energy'][i]:.4f}, ReconErr={history['recon_error'][i]:.4f}, AvgStd={history['avg_std_dev'][i]:.4f}"
            )
        if len(history["epoch"]) > idx:
            i = len(history["epoch"]) - 1
            print("...")
            print(
                f"Epoch {history['epoch'][i]}: FE={history['avg_free_energy'][i]:.4f}, ReconErr={history['recon_error'][i]:.4f}, AvgStd={history['avg_std_dev'][i]:.4f}"
            )

    print(
        f"\nFinal learned std devs (first 5): {grbm.get_std()[:5].detach().cpu().numpy()}"
    )

    print("\nChecking reconstruction on normalized data:")
    original_batch, _ = next(iter(dataloader))
    original_batch = (
        original_batch.view(original_batch.size(0), -1).float().to(grbm.device)
    )

    with torch.no_grad():
        reconstructed_batch_mean = grbm(original_batch)

    print(
        f"Original sample (first 5 values, first item):\n {original_batch[0, :5].cpu().numpy()}"
    )
    print(
        f"Reconstructed mean (first 5 values, first item):\n {reconstructed_batch_mean[0, :5].cpu().numpy()}"
    )
    print(
        f"MSE on batch: {F.mse_loss(reconstructed_batch_mean, original_batch).item():.4f}"
    )
