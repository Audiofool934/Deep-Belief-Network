import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import trange
from typing import Optional, Dict, List, Tuple, Union


def labels_to_one_hot(labels: torch.Tensor, n_classes: int) -> torch.Tensor:
    """Convert integer labels to one-hot vectors."""
    if labels.ndim == 0:  # Handle single label case
        labels = labels.unsqueeze(0)
    one_hot = F.one_hot(labels.long(), num_classes=n_classes)
    return one_hot.float()  # RBMs expect float input


# --- Base RBM Class ---
class RBM(nn.Module):
    """
    Restricted Boltzmann Machine (Bernoulli-Bernoulli) with optional label units.

    Models the joint probability P(features, labels, hidden).
    Labels are treated as additional binary visible units using one-hot encoding.
    """

    def __init__(
        self,
        n_visible_features: int,
        n_hidden: int,
        n_classes: int = 0,  # Number of classes for labeled data (0 if unsupervised)
        device: Optional[torch.device] = None,
    ) -> None:
        super(RBM, self).__init__()
        self.n_visible_features = n_visible_features
        self.n_hidden = n_hidden
        self.n_classes = n_classes
        self.n_visible_total = n_visible_features + n_classes

        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = device

        print(
            f"Initializing RBM: {n_visible_features} features, {n_classes} classes, "
            f"{n_hidden} hidden units on device: {self.device}"
        )

        # Parameters cover both features and label units
        self.W = nn.Parameter(
            torch.randn(n_hidden, self.n_visible_total, device=self.device) * 0.1
        )
        self.h_bias = nn.Parameter(torch.zeros(n_hidden, device=self.device))
        self.v_bias = nn.Parameter(
            torch.zeros(self.n_visible_total, device=self.device)
        )

        self.eps = 1e-8  # an epsilon of room for numerical stability

        self.to(self.device)

    def prepare_input_data(
        self, features: torch.Tensor, labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Ensures data is on the correct device, has the right type,
        and concatenates features and one-hot labels if provided.
        """
        features = features.float().to(self.device)
        if self.n_classes > 0:
            if labels is None:
                raise ValueError("Labels must be provided when n_classes > 0")
            if labels.shape[0] != features.shape[0]:
                raise ValueError(
                    f"Batch size mismatch between features ({features.shape[0]}) and labels ({labels.shape[0]})"
                )
            one_hot_labels = labels_to_one_hot(labels, self.n_classes).to(self.device)
            v = torch.cat((features, one_hot_labels), dim=1)
        else:
            if labels is not None:
                print(
                    "Warning: Labels provided but n_classes=0. Labels will be ignored."
                )
            v = features

        # Ensure the combined visible vector has the expected total dimension
        if v.shape[1] != self.n_visible_total:
            raise ValueError(
                f"Input dimension mismatch. Expected {self.n_visible_total}, got {v.shape[1]}"
            )

        return v

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given *all* visible units (features + labels).
        p(h=1 | v) = sigmoid( Wh*v + c )
        Assumes v already contains concatenated features and labels if applicable.
        """
        v = v.float().to(self.device)  # Ensure correct type and device
        prob_h = torch.sigmoid(F.linear(v, self.W, self.h_bias))
        return prob_h, torch.bernoulli(prob_h)

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample *all* visible units (features + labels) given hidden units.
        p(v=1 | h) = sigmoid( W.T*h + b )
        """
        h = h.float().to(self.device)
        prob_v = torch.sigmoid(F.linear(h, self.W.t(), self.v_bias))
        return prob_v, torch.bernoulli(prob_v)

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Perform one Gibbs sampling step (v -> h -> v'). Returns probabilities/means.
        Assumes v already contains concatenated features and labels if applicable.
        """
        v = v.float().to(self.device)
        prob_h, h_sample = self.sample_h(v)
        prob_v, v_sample = self.sample_v(h_sample)
        return prob_v  # Return probabilities for reconstruction checks

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy of a visible vector (features + labels).
        F(v) = - v*b - sum_j softplus( c_j + W_j*v )
        Assumes v already contains concatenated features and labels if applicable.
        """
        v = v.float().to(self.device)
        vbias_term = torch.matmul(
            v, self.v_bias
        )  # Equivalent to sum(v * v_bias) for binary
        wx_b = F.linear(v, self.W, self.h_bias)
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)
        return -vbias_term - hidden_term

    def contrastive_divergence(
        self, features: torch.Tensor, labels: Optional[torch.Tensor] = None, k: int = 1
    ) -> None:
        """
        Perform k-step Contrastive Divergence using Free Energy difference.
        Gradients are computed automatically via autograd.
        """
        v_0 = self.prepare_input_data(features, labels)

        v_k = v_0.clone()
        for _ in range(k):
            _, h_k = self.sample_h(v_k)
            _, v_k = self.sample_v(
                h_k
            )  # Note: v_k contains samples for features AND labels now
        v_k = v_k.detach()  # Stop gradients from flowing back through the Gibbs chain

        # We could optionally clamp the label part of v_k to v_0's labels here,
        # but standard CD often lets them evolve freely. Clamping might be
        # more appropriate for Persistent CD (PCD). Let's keep it simple for now.

        fe_positive = self.free_energy(v_0)
        fe_negative = self.free_energy(v_k)

        loss = torch.mean(fe_positive - fe_negative)
        loss.backward()

    def maximum_likelihood(
        self,
        features: torch.Tensor,
        labels: Optional[torch.Tensor] = None,
        mcmc_steps: int = 100,
    ) -> None:
        """
        Perform maximum likelihood estimation using MCMC sampling and Free Energy difference.
        Gradients are computed automatically via autograd.
        Note: True MLE is often intractable; this is an MCMC approximation.
        """
        v_data = self.prepare_input_data(features, labels)
        batch_size = v_data.size(0)

        # Initialize model samples
        with torch.no_grad():
            # Initialize based on visible biases
            initial_prob_v = torch.sigmoid(self.v_bias.repeat(batch_size, 1))
            v_model = torch.bernoulli(initial_prob_v).to(self.device)

            # Run MCMC chain
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
        classification_interval: int = 5,  # How often to check classification accuracy
    ) -> Optional[Dict[str, List[float]]]:
        """
        Train the RBM using selected method, autograd, and optionally monitor metrics.

        Args:
            data_loader: DataLoader yielding (features, labels) or just (features,)
                         if n_classes is 0.
            training_method: 'cd' or 'mle'.
            lr: Learning rate.
            k: Number of CD steps (if using CD).
            mcmc_steps: Number of MCMC steps (if using MLE).
            epochs: Number of training epochs.
            record_metrics: Whether to record training metrics (FE, Recon Err, Acc).
            weight_decay: L2 regularization parameter.
            classification_interval: Epoch interval to calculate classification accuracy (if n_classes > 0).

        Returns:
            Optional dictionary containing training metrics if record_metrics=True
        """
        optimizer = torch.optim.AdamW(
            self.parameters(), lr=lr, weight_decay=weight_decay
        )
        # optimizer = torch.optim.SGD(self.parameters(), lr=lr, weight_decay=weight_decay)
        training_method = training_method.lower()
        has_labels = self.n_classes > 0

        print(
            f"Training {self.__class__.__name__} using {training_method} with Autograd"
            f" (Labels: {'Yes' if has_labels else 'No'})"
        )
        training_loop = trange(epochs, desc="Initializing...")

        history: Dict[str, List[float]] = {
            "epoch": [],
            "avg_free_energy": [],
            "recon_error_features": [],
            "recon_error_labels": [],
            "classification_accuracy": [],
        }
        # Add GRBM specific metrics later if needed

        for epoch in training_loop:
            epoch_free_energy = 0.0
            epoch_recon_error_features = 0.0
            epoch_recon_error_labels = 0.0
            items_processed = 0
            correct_classifications = 0
            total_classified = 0

            self.train()  # Set model to training mode

            for batch in data_loader:
                # Unpack batch, handling both labeled and unlabeled cases
                if len(batch) == 2:
                    features, labels = batch
                    if not has_labels:  # Safety check
                        labels = None
                elif len(batch) == 1:
                    features = batch[0]
                    labels = None
                    if has_labels:
                        raise ValueError(
                            "DataLoader provided only features, but RBM expects labels (n_classes > 0)."
                        )
                else:
                    raise ValueError(
                        f"DataLoader yielded unexpected batch structure with {len(batch)} elements."
                    )

                features = features.view(
                    features.size(0), -1
                ).float()  # Flatten features
                current_batch_size = features.size(0)

                optimizer.zero_grad()

                # --- Perform Training Step ---
                if training_method == "cd":
                    self.contrastive_divergence(features, labels, k)
                elif training_method == "mle":
                    self.maximum_likelihood(features, labels, mcmc_steps)
                else:
                    raise ValueError(f"Unknown training method: {training_method}")

                # --- Gradient Clipping (Optional but recommended) ---
                torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

                # --- Check for invalid gradients ---
                valid_gradients = True
                for p in self.parameters():
                    if p.grad is not None:
                        if torch.isnan(p.grad).any() or torch.isinf(p.grad).any():
                            print(
                                f"\nWarning: NaN/Inf gradient detected in param {list(p.shape)} "
                                f"epoch {epoch+1}. Skipping update."
                            )
                            valid_gradients = False
                            break
                if not valid_gradients:
                    optimizer.zero_grad()  # Clear invalid gradients
                    continue  # Skip this batch update

                optimizer.step()

                # --- Record Metrics (Optional) ---
                if record_metrics:
                    with torch.no_grad():
                        self.eval()  # Switch to eval mode for consistent metrics
                        v_input = self.prepare_input_data(features, labels)

                        # Free Energy
                        current_free_energy = torch.mean(
                            self.free_energy(v_input)
                        ).item()
                        epoch_free_energy += current_free_energy * current_batch_size

                        # Reconstruction Error
                        v_reconstructed_prob = self.forward(v_input)
                        v_recon_features = v_reconstructed_prob[
                            :, : self.n_visible_features
                        ]
                        # Use MSE for features (even if binary, MSE works)
                        current_recon_error_features = F.mse_loss(
                            v_recon_features,
                            v_input[:, : self.n_visible_features],
                            reduction="mean",
                        ).item()
                        epoch_recon_error_features += (
                            current_recon_error_features * current_batch_size
                        )

                        if has_labels:
                            v_recon_labels = v_reconstructed_prob[
                                :, self.n_visible_features :
                            ]
                            # Use Cross-Entropy or MSE for label reconstruction (MSE is simpler here)
                            current_recon_error_labels = F.mse_loss(
                                v_recon_labels,
                                v_input[:, self.n_visible_features :],
                                reduction="mean",
                            ).item()
                            epoch_recon_error_labels += (
                                current_recon_error_labels * current_batch_size
                            )

                            # Classification accuracy (calculated periodically)
                            if (epoch + 1) % classification_interval == 0:
                                predicted_labels = self.classify(
                                    features
                                )  # Use the classify method
                                correct_classifications += (
                                    (predicted_labels.cpu() == labels.cpu())
                                    .sum()
                                    .item()
                                )
                                total_classified += current_batch_size
                        self.train()  # Switch back to train mode

                items_processed += current_batch_size

            # --- Log Epoch Metrics ---
            if record_metrics and items_processed > 0:
                avg_free_energy = epoch_free_energy / items_processed
                avg_recon_error_features = epoch_recon_error_features / items_processed

                history["epoch"].append(epoch + 1)
                history["avg_free_energy"].append(avg_free_energy)
                history["recon_error_features"].append(avg_recon_error_features)

                desc = (
                    f"Epoch {epoch+1}/{epochs}, Avg FE: {avg_free_energy:.4f}, "
                    f"ReconErr(Feat): {avg_recon_error_features:.4f}"
                )

                if has_labels:
                    avg_recon_error_labels = epoch_recon_error_labels / items_processed
                    history["recon_error_labels"].append(avg_recon_error_labels)
                    desc += f", ReconErr(Lbl): {avg_recon_error_labels:.4f}"

                    if total_classified > 0:  # Means classification was run this epoch
                        accuracy = (correct_classifications / total_classified) * 100
                        history["classification_accuracy"].append(accuracy)
                        desc += f", Acc: {accuracy:.2f}%"
                    else:  # Append placeholder if not calculated this epoch
                        history["classification_accuracy"].append(
                            float("nan")
                        )  # Or keep previous value

                training_loop.set_description(desc)

            elif items_processed > 0:
                training_loop.set_description(f"Epoch {epoch+1}/{epochs} Completed")
            else:
                training_loop.set_description(
                    f"Epoch {epoch+1}/{epochs} - No items processed"
                )

        print("Training finished.")
        # Clean up history for metrics not always calculated
        if not has_labels:
            del history["recon_error_labels"]
            del history["classification_accuracy"]
        elif any(torch.isnan(torch.tensor(history["classification_accuracy"]))):
            print("\nNote: Classification accuracy was calculated periodically.")

        return history

    # --- New Methods for Generation and Classification ---

    @torch.no_grad()
    def generate(
        self,
        target_label: Union[int, torch.Tensor],
        n_samples: int = 1,
        gibbs_steps: int = 200,
    ) -> torch.Tensor:
        """
        Generate feature samples conditioned on a specific target label.

        Args:
            target_label: The integer class label to condition on.
            n_samples: Number of samples to generate.
            gibbs_steps: Number of Gibbs sampling steps for generation.

        Returns:
            Tensor of generated feature samples of shape [n_samples, n_visible_features].
        """
        if not (0 <= target_label < self.n_classes):
            raise ValueError(
                f"target_label {target_label} is out of bounds for {self.n_classes} classes."
            )
        if self.n_classes == 0:
            raise RuntimeError("Cannot generate by label if n_classes is 0.")

        self.eval()  # Set to evaluation mode

        # Prepare the fixed label part
        label_tensor = torch.tensor([target_label] * n_samples, device=self.device)
        target_one_hot = labels_to_one_hot(label_tensor, self.n_classes).to(self.device)

        # Initialize visible state (random features, fixed labels)
        # Initialize features based on bias or randomly
        feature_probs = torch.sigmoid(self.v_bias[: self.n_visible_features])
        v_features = torch.bernoulli(feature_probs.repeat(n_samples, 1))
        # For GRBM, initialize with Gaussian noise around bias (see override)

        v = torch.cat((v_features, target_one_hot), dim=1)

        # Run Gibbs chain with clamped labels
        for _ in range(gibbs_steps):
            _, h_samples = self.sample_h(v)
            v_probs, v_samples = self.sample_v(h_samples)

            # Update features based on sampling, but keep labels clamped
            v = torch.cat(
                (v_samples[:, : self.n_visible_features], target_one_hot), dim=1
            )
            # Alternative: Use probabilities for smoother generation
            # v = torch.cat(
            #     (v_probs[:, : self.n_visible_features], target_one_hot), dim=1
            # )

        # Return only the feature part of the final visible state
        return v[:, : self.n_visible_features].cpu()

    @torch.no_grad()
    def classify(self, features: torch.Tensor) -> torch.Tensor:
        """
        Classify input features by finding the label that minimizes free energy.

        Args:
            features: Tensor of feature vectors [batch_size, n_visible_features].

        Returns:
            Tensor of predicted integer labels [batch_size].
        """
        if self.n_classes == 0:
            raise RuntimeError("Cannot classify if n_classes is 0.")

        self.eval()  # Set to evaluation mode
        features = features.view(features.size(0), -1).float().to(self.device)
        batch_size = features.shape[0]

        min_fe = torch.full((batch_size,), float("inf"), device=self.device)
        predicted_labels = torch.zeros(batch_size, dtype=torch.long, device=self.device)

        # Iterate through each possible class
        for label_idx in range(self.n_classes):
            # Create one-hot vectors for this class for the whole batch
            current_label = torch.tensor([label_idx] * batch_size, device=self.device)
            one_hot = labels_to_one_hot(current_label, self.n_classes).to(self.device)

            # Combine features and current candidate label
            v_candidate = torch.cat((features, one_hot), dim=1)

            # Calculate free energy for this feature-label combination
            fe = self.free_energy(v_candidate)

            # Update predictions if this label yields lower free energy
            update_mask = fe < min_fe
            min_fe[update_mask] = fe[update_mask]
            predicted_labels[update_mask] = label_idx

        return predicted_labels.cpu()


# --- Gaussian-Bernoulli RBM (Modified for Labels) ---
class GaussianBernoulliRBM(RBM):
    """
    Gaussian-Bernoulli RBM with optional label units.

    Models P(features, labels, hidden) where features are continuous (Gaussian)
    and labels (one-hot) and hidden units are binary (Bernoulli).
    Includes learnable standard deviation for feature units.
    """

    def __init__(
        self,
        n_visible_features: int,
        n_hidden: int,
        n_classes: int = 0,
        device: Optional[torch.device] = None,
        learn_std: bool = True,  # Whether std dev should be learned
        initial_std: float = 1.0,  # Initial std dev if learned, or fixed value if not
    ) -> None:
        super(GaussianBernoulliRBM, self).__init__(
            n_visible_features, n_hidden, n_classes, device
        )

        # Standard deviation parameter ONLY for the feature part
        if learn_std:
            # Initialize log_std around log(initial_std)
            initial_log_std = torch.log(torch.tensor(initial_std, device=self.device))
            self.log_std_features = nn.Parameter(
                torch.full(
                    (n_visible_features,), initial_log_std.item(), device=self.device
                )
                # torch.zeros(n_visible_features, device=self.device) # Alt: Init near std=1
            )
            print(
                f"GRBM initialized with LEARNABLE std dev (initial ~{initial_std:.3f})."
            )
        else:
            # Fixed standard deviation (register as buffer, not parameter)
            self.register_buffer(
                "log_std_features",
                torch.full(
                    (n_visible_features,),
                    torch.log(torch.tensor(initial_std)).item(),
                    device=self.device,
                ),
            )
            print(f"GRBM initialized with FIXED std dev ({initial_std:.3f}).")

        self.learn_std = learn_std

        # Re-initialize weight matrix with potentially smaller values for stability with Gaussian units
        # stdv = 1. / torch.sqrt(torch.tensor(self.n_visible_features + self.n_hidden)) # Xavier/Glorot init idea
        # self.W.data.uniform_(-stdv, stdv)
        self.W.data *= 0.05  # Often helps stability

        print(
            f"Instantiated GaussianBernoulliRBM ({'learnable std' if learn_std else 'fixed std'})."
        )

    def get_std_features(self) -> torch.Tensor:
        """Returns the standard deviation for feature units."""
        return torch.exp(self.log_std_features)

    def get_var_features(self) -> torch.Tensor:
        """Returns the variance for feature units. Clamped for stability."""
        # Clamp log_std to prevent huge/tiny variances
        # clamped_log_std = torch.clamp(self.log_std_features, -5.0, 5.0) # Prevents exp overflow/underflow
        # return torch.exp(2 * clamped_log_std) + self.eps
        return (
            torch.exp(2 * self.log_std_features) + self.eps
        )  # Add eps AFTER exponentiation

    def sample_h(self, v: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample hidden units given visible units (Gaussian features + Binary labels).
        p(h_j=1 | v) = sigmoid( c_j + sum_i (W_ji * v_i / var_i) + sum_k (W_jk * y_k) )
        Assumes v contains concatenated features and labels.
        """
        v = v.float().to(self.device)
        v_features = v[:, : self.n_visible_features]
        v_labels = v[:, self.n_visible_features :]  # Empty if n_classes=0

        # Get weights corresponding to features and labels
        W_features = self.W[:, : self.n_visible_features]
        W_labels = self.W[:, self.n_visible_features :]  # Empty if n_classes=0

        # Scale features by their variance
        var_f = self.get_var_features()
        v_features_scaled = v_features / var_f  # Shape: [batch, n_features]

        # Calculate activation from features and labels
        feature_activation = F.linear(
            v_features_scaled, W_features
        )  # [batch, n_hidden]
        label_activation = torch.zeros_like(feature_activation)
        if self.n_classes > 0:
            label_activation = F.linear(v_labels, W_labels)  # [batch, n_hidden]

        total_activation = feature_activation + label_activation + self.h_bias
        prob_h = torch.sigmoid(total_activation)
        return prob_h, torch.bernoulli(prob_h)

    def sample_v(self, h: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Sample visible units (Gaussian features + Binary labels) given hidden units.
        v_features ~ N( mean = b_f + W_f.T @ h, variance = var_f )
        p(v_labels=1 | h) = sigmoid( b_l + W_l.T @ h )
        """
        h = h.float().to(self.device)

        # Get weights corresponding to features and labels
        W_features_t = self.W[
            :, : self.n_visible_features
        ].t()  # [n_features, n_hidden]
        W_labels_t = self.W[:, self.n_visible_features :].t()  # [n_labels, n_hidden]

        # Get biases for features and labels
        v_bias_features = self.v_bias[: self.n_visible_features]
        v_bias_labels = self.v_bias[self.n_visible_features :]

        # --- Sample Features (Gaussian) ---
        mean_v_features = F.linear(
            h, W_features_t, v_bias_features
        )  # [batch, n_features]
        std_dev_f = self.get_std_features()
        # Expand std_dev to batch size
        std_dev_f_batch = std_dev_f.expand_as(mean_v_features)
        # Sample from Gaussian
        noise = torch.randn_like(mean_v_features, device=self.device)
        v_sample_features = mean_v_features + noise * std_dev_f_batch

        # --- Sample Labels (Bernoulli) ---
        if self.n_classes > 0:
            logits_v_labels = F.linear(
                h, W_labels_t, v_bias_labels
            )  # [batch, n_labels]
            prob_v_labels = torch.sigmoid(logits_v_labels)
            v_sample_labels = torch.bernoulli(prob_v_labels)
        else:
            # No labels to sample
            prob_v_labels = torch.empty((h.shape[0], 0), device=self.device)
            v_sample_labels = torch.empty((h.shape[0], 0), device=self.device)

        # Concatenate probabilities/means and samples
        v_prob_or_mean = torch.cat((mean_v_features, prob_v_labels), dim=1)
        v_sample = torch.cat((v_sample_features, v_sample_labels), dim=1)

        return v_prob_or_mean, v_sample

    def forward(self, v: torch.Tensor) -> torch.Tensor:
        """
        Perform one Gibbs sampling step (v -> h -> v'). Returns means/probabilities.
        Assumes v contains concatenated features and labels if applicable.
        """
        v = v.float().to(self.device)
        prob_h, h_sample = self.sample_h(v)
        v_mean_or_prob, v_sample_recon = self.sample_v(h_sample)
        return v_mean_or_prob  # Return feature means and label probabilities

    def free_energy(self, v: torch.Tensor) -> torch.Tensor:
        """
        Compute the free energy for mixed visible units (Gaussian features + Binary labels).
        F(v) = sum_i (v_i - b_i)^2 / (2*var_i)  <-- Feature part (Gaussian)
               - sum_k y_k * b_k              <-- Label part (Binary)
               - sum_j softplus(...)          <-- Hidden part (depends on both)
        Assumes v contains concatenated features and labels.
        """
        v = v.float().to(self.device)
        v_features = v[:, : self.n_visible_features]
        v_labels = v[:, self.n_visible_features :]  # Empty if n_classes=0

        # Biases
        v_bias_features = self.v_bias[: self.n_visible_features]
        v_bias_labels = self.v_bias[self.n_visible_features :]

        # --- Visible Term Calculation ---
        # Gaussian part (Features)
        var_f = self.get_var_features()
        visible_term_features = torch.sum(
            0.5 * ((v_features - v_bias_features) ** 2) / var_f, dim=1
        )

        # Binary part (Labels)
        visible_term_labels = torch.zeros_like(visible_term_features)
        if self.n_classes > 0:
            # Note: The standard binary RBM FE is -v*b - softplus(...).
            # Here we group terms differently. Contribution from visible bias for binary units is -v*b.
            visible_term_labels = -torch.matmul(
                v_labels, v_bias_labels
            )  # Shape: [batch]

        total_visible_term = visible_term_features + visible_term_labels

        # --- Hidden Term Calculation ---
        # Uses the same logic as in sample_h activation
        W_features = self.W[:, : self.n_visible_features]
        W_labels = self.W[:, self.n_visible_features :]
        v_features_scaled = v_features / var_f

        feature_activation = F.linear(v_features_scaled, W_features)
        label_activation = torch.zeros_like(feature_activation)
        if self.n_classes > 0:
            label_activation = F.linear(v_labels, W_labels)

        wx_b = feature_activation + label_activation + self.h_bias
        hidden_term = torch.sum(F.softplus(wx_b), dim=1)  # Sum over hidden units

        # Combine terms: F(v) = VisibleEnergy - HiddenTermContribution
        # Standard definition often uses F(v) = -log(sum_h exp(-E(v,h)))
        # Which leads to the form: VisibleTerm - HiddenTerm (as derived for GBRBM/RBM)
        return total_visible_term - hidden_term

    # Override train_rbm to add GRBM specific metrics if needed
    def train_rbm(self, *args, **kwargs) -> Optional[Dict[str, List[float]]]:
        # Call the base class train_rbm
        history = super().train_rbm(*args, **kwargs)

        # Add GRBM specific metrics to history if desired
        if history and self.learn_std:
            # Example: track average learned std dev over epochs
            # This requires running inference again or storing batch std devs
            # For simplicity, let's calculate it at the end of training here
            avg_std_dev_per_epoch = []
            # Need to re-implement metric collection within the loop for this...
            # Or just report the final std dev.
            final_avg_std = torch.mean(self.get_std_features()).item()
            print(f"Final average learned feature std dev: {final_avg_std:.4f}")
            # history['avg_std_dev'] = [...] # Add if collected per epoch

        return history

    # Override generate for GRBM feature initialization
    @torch.no_grad()
    def generate(
        self,
        target_label: Union[int, torch.Tensor],
        n_samples: int = 1,
        gibbs_steps: int = 200,
    ) -> torch.Tensor:
        """
        Generate feature samples conditioned on a specific target label (GRBM version).
        Initializes features using Gaussian noise around feature biases.
        """
        if not (0 <= target_label < self.n_classes):
            raise ValueError(
                f"target_label {target_label} is out of bounds for {self.n_classes} classes."
            )
        if self.n_classes == 0:
            raise RuntimeError("Cannot generate by label if n_classes is 0.")

        self.eval()

        label_tensor = torch.tensor([target_label] * n_samples, device=self.device)
        target_one_hot = labels_to_one_hot(label_tensor, self.n_classes).to(self.device)

        # Initialize features with noise N(bias, std^2)
        feature_biases = self.v_bias[: self.n_visible_features].repeat(n_samples, 1)
        feature_stds = self.get_std_features().repeat(n_samples, 1)
        v_features = torch.normal(mean=feature_biases, std=feature_stds)

        v = torch.cat((v_features, target_one_hot), dim=1)

        # Run Gibbs chain with clamped labels
        for _ in range(gibbs_steps):
            _, h_samples = self.sample_h(v)
            # Use the sample_v method which handles Gaussian features and Binary labels
            v_mean_or_prob, v_samples = self.sample_v(h_samples)

            # Update features based on sampling, keep labels clamped
            v = torch.cat(
                (v_samples[:, : self.n_visible_features], target_one_hot), dim=1
            )
            # Alternative: Use mean for features (smoother)
            # v = torch.cat((v_mean_or_prob[:, :self.n_visible_features], target_one_hot), dim=1)

        # Return only the feature part
        return v[:, : self.n_visible_features].cpu()


# --- Example Usage (MNIST) ---
if __name__ == "__main__":
    import torch.utils.data as data
    from torchvision.datasets import MNIST
    from torchvision.transforms import ToTensor
    import matplotlib.pyplot as plt
    import numpy as np
    from sklearn.preprocessing import StandardScaler

    BATCH_SIZE = 128
    N_FEATURES = 28 * 28  # MNIST image size
    N_HIDDEN = 256
    N_CLASSES = 10
    EPOCHS = 50  # Keep low for quick demo
    LR = 0.005  # AdamW often works better than SGD for RBMs
    K_CD = 1
    CLASSIFY_INTERVAL = 2  # Check accuracy every 2 epochs

    # --- Data Loading ---
    print("\nLoading MNIST data...")
    try:
        train_dataset = MNIST(
            root="./data", train=True, download=True, transform=ToTensor()
        )
        test_dataset = MNIST(
            root="./data", train=False, download=True, transform=ToTensor()
        )

        # --- Choose Model Type ---
        MODEL_TYPE = "GRBM"  # Change to "RBM" for Bernoulli-Bernoulli

        if MODEL_TYPE == "RBM":
            # Binarize MNIST for standard RBM
            print("Binarizing data for standard RBM...")
            train_features = (
                train_dataset.data.view(-1, N_FEATURES).float() / 255.0 > 0.5
            ).float()
            test_features = (
                test_dataset.data.view(-1, N_FEATURES).float() / 255.0 > 0.5
            ).float()
            train_labels = train_dataset.targets
            test_labels = test_dataset.targets
            learn_std_grbm = False  # Not applicable

        elif MODEL_TYPE == "GRBM":
            print("Normalizing data for GRBM...")
            # Use normalized continuous data for GRBM
            scaler = StandardScaler()
            train_features_flat = (
                train_dataset.data.view(-1, N_FEATURES).float().numpy()
            )
            test_features_flat = test_dataset.data.view(-1, N_FEATURES).float().numpy()

            # Fit on training data ONLY
            scaler.fit(train_features_flat)
            train_features = torch.tensor(
                scaler.transform(train_features_flat), dtype=torch.float32
            )
            test_features = torch.tensor(
                scaler.transform(test_features_flat), dtype=torch.float32
            )

            train_labels = train_dataset.targets
            test_labels = test_dataset.targets
            learn_std_grbm = True  # Learn the std dev
            initial_std_grbm = 1.0  # Start near 1 since data is normalized

        else:
            raise ValueError("MODEL_TYPE must be 'RBM' or 'GRBM'")

        # Create TensorDatasets and DataLoaders
        train_torch_dataset = data.TensorDataset(train_features, train_labels)
        test_torch_dataset = data.TensorDataset(test_features, test_labels)

        train_loader = data.DataLoader(
            train_torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=2,
            pin_memory=True,
        )
        test_loader = data.DataLoader(
            test_torch_dataset,
            batch_size=BATCH_SIZE,
            shuffle=False,
            num_workers=2,
            pin_memory=True,
        )

        print(
            f"Data loaded: {len(train_dataset)} train, {len(test_dataset)} test samples."
        )
        print(f"Feature shape per sample: {train_features.shape[1:]}")

        # --- Model Instantiation ---
        print(f"\n--- Testing {MODEL_TYPE} with Labels ---")
        if MODEL_TYPE == "RBM":
            rbm_model = RBM(
                n_visible_features=N_FEATURES, n_hidden=N_HIDDEN, n_classes=N_CLASSES
            )
        else:  # GRBM
            rbm_model = GaussianBernoulliRBM(
                n_visible_features=N_FEATURES,
                n_hidden=N_HIDDEN,
                n_classes=N_CLASSES,
                learn_std=learn_std_grbm,
                initial_std=initial_std_grbm,
            )

        print(f"Model parameters on device: {next(rbm_model.parameters()).device}")

        # --- Training ---
        history = rbm_model.train_rbm(
            train_loader,
            training_method="cd",  # CD is generally faster/more stable
            lr=LR,
            k=K_CD,
            epochs=EPOCHS,
            record_metrics=True,
            weight_decay=1e-4,
            classification_interval=CLASSIFY_INTERVAL,
        )

        # --- Evaluation ---
        if history:
            print("\nTraining History (Sample):")
            keys = list(history.keys())
            num_epochs_recorded = len(history["epoch"])
            indices_to_print = list(range(min(3, num_epochs_recorded))) + list(
                range(max(0, num_epochs_recorded - 2), num_epochs_recorded)
            )
            indices_to_print = sorted(
                list(set(indices_to_print))
            )  # Unique sorted indices

            for i in indices_to_print:
                log_str = f"Epoch {history['epoch'][i]}: "
                log_str += ", ".join(
                    [
                        f"{key}={history[key][i]:.4f}"
                        for key in keys
                        if key != "epoch" and not np.isnan(history[key][i])
                    ]
                )
                print(log_str)

        # Test Classification Accuracy on Test Set
        print("\nCalculating final classification accuracy on Test Set...")
        correct_predictions = 0
        total_samples = 0
        rbm_model.eval()
        with torch.no_grad():
            for features, labels in test_loader:
                features = features.view(features.size(0), -1)  # Flatten if needed
                predicted_labels = rbm_model.classify(features)
                correct_predictions += (
                    (predicted_labels.cpu() == labels.cpu()).sum().item()
                )
                total_samples += labels.size(0)

        if total_samples > 0:
            test_accuracy = (correct_predictions / total_samples) * 100
            print(f"Test Set Classification Accuracy: {test_accuracy:.2f}%")
        else:
            print("Could not calculate test accuracy (no samples processed).")

        # --- Generation ---
        print("\nGenerating samples for each digit (0-9)...")
        n_generate_per_class = 5
        fig, axes = plt.subplots(
            N_CLASSES,
            n_generate_per_class,
            figsize=(n_generate_per_class * 1.5, N_CLASSES * 1.5),
        )
        fig.suptitle(f"{MODEL_TYPE} Generated Samples by Label", fontsize=16)

        for digit in range(N_CLASSES):
            generated_features = rbm_model.generate(
                target_label=digit,
                n_samples=n_generate_per_class,
                gibbs_steps=1000,  # More steps for better samples
            )

            # Inverse transform for GRBM if needed to visualize
            if MODEL_TYPE == "GRBM" and "scaler" in locals():
                generated_features = torch.tensor(
                    scaler.inverse_transform(generated_features.numpy()),
                    dtype=torch.float32,
                )

            for i in range(n_generate_per_class):
                ax = axes[digit, i]
                img = generated_features[i].view(28, 28).cpu().numpy()
                ax.imshow(img, cmap="gray")
                ax.set_title(f"Gen: {digit}")
                ax.axis("off")

        plt.tight_layout(
            rect=[0, 0.03, 1, 0.95]
        )  # Adjust layout to prevent title overlap
        plt.savefig(f"{MODEL_TYPE}_generated_samples.png")
        print(f"Generated samples saved to {MODEL_TYPE}_generated_samples.png")
        # plt.show() # Uncomment to display plot directly

    except ImportError:
        print("\nMNIST dataset not found or torchvision not installed.")
        print("Please install torchvision: pip install torchvision")
    except Exception as e:
        print(f"\nAn error occurred during the example execution: {e}")
        import traceback

        traceback.print_exc()
