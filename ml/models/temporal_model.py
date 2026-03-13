"""
GRU Temporal Tilt Model — Stage 3 of the ML pipeline.

Takes a *sequence* of feature vectors (one per snapshot, 5-second intervals)
and outputs P(tilted) as a real-time posterior that updates with each new
snapshot.

=== WHY A GRU (NOT LSTM OR TRANSFORMER)? ===

For our problem:
  - Sequences are short: 6–24 snapshots = 30 seconds to 2 minutes
  - Dataset will be small: ~500–2000 labeled game sequences
  - We need the model to run at inference time in < 10ms on CPU

A Transformer needs more data to shine (attention learns from many sequences).
An LSTM is comparable to GRU but slightly more parameters with no benefit here.
A GRU (Gated Recurrent Unit) is the right balance:
  - Captures temporal dependencies via its reset and update gates
  - Lighter than LSTM (fewer parameters → less overfitting on small data)
  - Fast at inference (one forward pass per new snapshot)

=== MATHEMATICAL FORMULATION ===

At time t, given the hidden state h_{t-1} and input x_t ∈ R^27:

  Update gate:  z_t = σ(W_z x_t + U_z h_{t-1} + b_z)
  Reset gate:   r_t = σ(W_r x_t + U_r h_{t-1} + b_r)
  Candidate:    h̃_t = tanh(W_h x_t + U_h (r_t ⊙ h_{t-1}) + b_h)
  Hidden:       h_t = (1 - z_t) ⊙ h_{t-1} + z_t ⊙ h̃_t

  Output: p_t = σ(W_out h_t + b_out)  →  P(tilted at time t)

The update gate z_t controls how much new information to incorporate.
When the game is calm (no kill events, stable CS), z_t stays near 0 and
the model retains its prior estimate. When sudden bad events occur (multiple
deaths, item sell), z_t spikes toward 1 and the estimate updates quickly.

=== SEQUENCE LABELING STRATEGY ===

Training uses the per-game outcome label (performed_poorly) as the target
for ALL snapshots in the game. This is the "last-state supervision" approach:

  Loss = CrossEntropy(p_T, y_game)   [T = last snapshot]

We don't try to label individual snapshots (we don't know exactly when tilt
started). The model learns to predict the game outcome from an increasing
amount of evidence.

An alternative (for future work) is "sequence labeling" where we use the
TiltPredictionLog's peak_time to create a ramp label: y_t = 0 before peak,
1 after. This requires more engineering and sufficient data.

=== INFERENCE IN THE LIVE PIPELINE ===

The GRU is stateful at inference time:
  - At game start, h_0 = 0
  - Every 5 seconds, call model.step(x_t) → p_t
  - h_t is retained internally between calls
  - No need to reprocess the full sequence each time (O(1) per snapshot)

This makes real-time use lightweight: one GRU forward pass ≈ 0.2ms on CPU.
"""

from __future__ import annotations

import logging
import pickle
from pathlib import Path
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not installed — TemporalTiltModel unavailable.")

from ml.features.feature_extractor import FEATURE_DIM


# ── GRU Module ────────────────────────────────────────────────────────────────

if TORCH_AVAILABLE:

    class TiltGRU(nn.Module):
        """
        Two-layer GRU binary classifier for temporal tilt detection.

        Architecture:
          Input (FEATURE_DIM=27)
            → GRU layer 1 (hidden_dim, dropout between layers)
            → GRU layer 2 (hidden_dim)
            → LayerNorm
            → Linear(hidden_dim → 1)
            → Sigmoid → P(tilted)

        hidden_dim=64 with 2 layers gives ~35k parameters — appropriate for
        a dataset of 500–2000 sequences. Larger models would overfit.
        """

        def __init__(
            self,
            input_dim:  int = FEATURE_DIM,
            hidden_dim: int = 64,
            n_layers:   int = 2,
            dropout:    float = 0.30,
        ):
            super().__init__()
            self.hidden_dim = hidden_dim
            self.n_layers   = n_layers

            self.gru = nn.GRU(
                input_size=input_dim,
                hidden_size=hidden_dim,
                num_layers=n_layers,
                batch_first=True,
                dropout=dropout if n_layers > 1 else 0.0,
            )
            self.norm = nn.LayerNorm(hidden_dim)
            self.head = nn.Sequential(
                nn.Linear(hidden_dim, 32),
                nn.ReLU(),
                nn.Dropout(p=dropout),
                nn.Linear(32, 1),
                nn.Sigmoid(),
            )

        def forward(self, x: "torch.Tensor", h0: "torch.Tensor | None" = None):
            """
            Args:
                x:  shape (batch, seq_len, FEATURE_DIM)
                h0: initial hidden state shape (n_layers, batch, hidden_dim), or None

            Returns:
                out: shape (batch, seq_len, 1) — P(tilted) at each time step
                h_n: final hidden state (n_layers, batch, hidden_dim)
            """
            gru_out, h_n = self.gru(x, h0)          # (batch, seq, hidden)
            gru_out      = self.norm(gru_out)
            out          = self.head(gru_out)         # (batch, seq, 1)
            return out, h_n

        def step(self, x_t: "torch.Tensor", h: "torch.Tensor | None" = None):
            """
            Single-step inference for real-time use.

            Args:
                x_t: shape (1, FEATURE_DIM) — current snapshot feature vector
                h:   current hidden state (n_layers, 1, hidden_dim) or None

            Returns:
                p:  float — P(tilted) at this timestep
                h:  updated hidden state
            """
            x_t = x_t.unsqueeze(0)  # → (1, 1, FEATURE_DIM) for batch_first GRU
            out, h = self.forward(x_t, h)
            p = float(out[0, -1, 0].item())
            return p, h


# ── Wrapper class ─────────────────────────────────────────────────────────────

class TemporalTiltModel:
    """
    Training + inference wrapper around TiltGRU.

    Usage (training):
        model = TemporalTiltModel()
        metrics = model.train(sequences, labels)
        model.save("experiments/temporal_model.pt")

    Usage (live inference):
        model = TemporalTiltModel.load("experiments/temporal_model.pt")
        model.reset_state("PlayerName")
        for snapshot in game_snapshots:
            p, _ = model.predict_step(feature_vector, "PlayerName")
            print(f"P(tilted) = {p:.2f}")
    """

    DEFAULT_HYPERPARAMS = {
        "hidden_dim": 64,
        "n_layers":   2,
        "dropout":    0.30,
        "lr":         1e-3,
        "weight_decay": 1e-4,
        "epochs":     100,
        "patience":   15,       # early stopping patience
        "batch_size": 32,
    }

    def __init__(self, hyperparams: dict | None = None):
        if not TORCH_AVAILABLE:
            raise ImportError("PyTorch is required. It's already installed in your conda env.")
        self.hp      = hyperparams or self.DEFAULT_HYPERPARAMS
        self._model: Any = None
        self._device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.is_fitted = False
        # Per-player hidden state for live inference
        self._live_states: dict[str, Any] = {}

    def train(
        self,
        sequences: list[np.ndarray],
        labels:    list[int],
        val_sequences: list[np.ndarray] | None = None,
        val_labels:    list[int]        | None = None,
    ) -> dict[str, list[float]]:
        """
        Train the GRU model on a list of per-player game sequences.

        Args:
            sequences: list of np.ndarray, each shape (T_i, FEATURE_DIM) — variable length.
                       T_i is the number of snapshots in game i.
            labels:    list of int (0 or 1) — one per game sequence.
            val_sequences: optional validation sequences
            val_labels:    optional validation labels

        Returns:
            dict with "train_loss" and "val_loss" histories (list per epoch)
        """
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader

        self._model = TiltGRU(
            input_dim=FEATURE_DIM,
            hidden_dim=self.hp["hidden_dim"],
            n_layers=self.hp["n_layers"],
            dropout=self.hp["dropout"],
        ).to(self._device)

        optimizer = torch.optim.AdamW(
            self._model.parameters(),
            lr=self.hp["lr"],
            weight_decay=self.hp["weight_decay"],
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=0.5, patience=5,
        )

        # Compute class weight for imbalanced data
        pos_rate = sum(labels) / max(len(labels), 1)
        pos_weight = torch.tensor([(1 - pos_rate) / max(pos_rate, 1e-9)],
                                  dtype=torch.float32, device=self._device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        history: dict[str, list[float]] = {"train_loss": [], "val_loss": []}
        best_val_loss = float("inf")
        best_state    = None
        patience_counter = 0

        n_epochs = self.hp["epochs"]
        logger.info(
            f"Training TiltGRU on {len(sequences)} sequences "
            f"({sum(labels)} positive, {pos_rate:.1%} base rate), "
            f"device={self._device}"
        )

        for epoch in range(n_epochs):
            self._model.train()
            epoch_loss = 0.0

            # Shuffle training data (simple random permutation — no DataLoader needed
            # for variable-length sequences which we process one at a time)
            indices = np.random.permutation(len(sequences))
            optimizer.zero_grad()
            batch_loss = torch.tensor(0.0, device=self._device)
            batch_count = 0

            for i, idx in enumerate(indices):
                seq = torch.tensor(sequences[idx], dtype=torch.float32, device=self._device)
                seq = seq.unsqueeze(0)   # (1, T, FEATURE_DIM)
                y   = torch.tensor([[labels[idx]]], dtype=torch.float32, device=self._device)

                out, _ = self._model(seq)  # (1, T, 1)
                # Use LAST timestep for classification loss
                logit = _sigmoid_to_logit(out[0, -1, 0].unsqueeze(0).unsqueeze(0))
                loss  = criterion(logit, y)
                batch_loss  = batch_loss + loss
                batch_count += 1
                epoch_loss  += loss.item()

                # Mini-batch gradient accumulation
                if batch_count >= self.hp["batch_size"] or i == len(indices) - 1:
                    (batch_loss / batch_count).backward()
                    nn.utils.clip_grad_norm_(self._model.parameters(), max_norm=1.0)
                    optimizer.step()
                    optimizer.zero_grad()
                    batch_loss  = torch.tensor(0.0, device=self._device)
                    batch_count = 0

            avg_train_loss = epoch_loss / max(len(sequences), 1)
            history["train_loss"].append(round(avg_train_loss, 5))

            # ── Validation ───────────────────────────────────────────────────
            val_loss = None
            if val_sequences is not None and val_labels is not None:
                val_loss = self._eval_loss(val_sequences, val_labels, criterion)
                history["val_loss"].append(round(val_loss, 5))
                scheduler.step(val_loss)

                if val_loss < best_val_loss:
                    best_val_loss    = val_loss
                    best_state       = {k: v.clone() for k, v in self._model.state_dict().items()}
                    patience_counter = 0
                else:
                    patience_counter += 1

                if patience_counter >= self.hp["patience"]:
                    logger.info(f"Early stopping at epoch {epoch+1}")
                    break

            if (epoch + 1) % 10 == 0:
                val_str = f", val_loss={val_loss:.4f}" if val_loss is not None else ""
                logger.info(f"Epoch {epoch+1}/{n_epochs}  train_loss={avg_train_loss:.4f}{val_str}")

        # Restore best model if validation was used
        if best_state is not None:
            self._model.load_state_dict(best_state)
            logger.info(f"Restored best model (val_loss={best_val_loss:.4f})")

        self.is_fitted = True
        return history

    def predict_proba_sequence(self, sequence: np.ndarray) -> np.ndarray:
        """
        Score a full sequence at once (batch mode for evaluation).

        Args:
            sequence: shape (T, FEATURE_DIM)

        Returns:
            np.ndarray shape (T,) — P(tilted) at each timestep
        """
        self._check_fitted()
        import torch

        self._model.eval()
        with torch.no_grad():
            x   = torch.tensor(sequence, dtype=torch.float32, device=self._device)
            x   = x.unsqueeze(0)    # (1, T, FEATURE_DIM)
            out, _ = self._model(x)  # (1, T, 1)
            return out[0, :, 0].cpu().numpy()

    def predict_step(
        self,
        feature_vector: np.ndarray,
        player_name: str,
    ) -> tuple[float, dict]:
        """
        Real-time single-step inference. Maintains hidden state per player.

        Call reset_state(player_name) at game start.

        Args:
            feature_vector: shape (FEATURE_DIM,) — current snapshot features
            player_name:    used as key for hidden state storage

        Returns:
            (p_tilted: float, state_info: dict)
        """
        self._check_fitted()
        import torch

        self._model.eval()
        with torch.no_grad():
            x = torch.tensor(feature_vector, dtype=torch.float32, device=self._device)
            x = x.unsqueeze(0)   # (1, FEATURE_DIM)
            h = self._live_states.get(player_name)
            p, h_new = self._model.step(x, h)
            self._live_states[player_name] = h_new

        return p, {"hidden_state_norm": float(h_new.norm().item())}

    def reset_state(self, player_name: str) -> None:
        """Reset hidden state for a player (call at game start)."""
        self._live_states.pop(player_name, None)

    def reset_all_states(self) -> None:
        """Reset all live hidden states (call at game_over)."""
        self._live_states.clear()

    def save(self, path: str | Path) -> None:
        self._check_fitted()
        import torch

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            "model_state": self._model.state_dict(),
            "hyperparams":  self.hp,
            "feature_dim":  FEATURE_DIM,
        }, path)
        logger.info(f"TemporalTiltModel saved to {path}")

    @classmethod
    def load(cls, path: str | Path) -> "TemporalTiltModel":
        import torch

        data = torch.load(path, map_location="cpu")
        model = cls(hyperparams=data["hyperparams"])
        model._model = TiltGRU(
            input_dim=data.get("feature_dim", FEATURE_DIM),
            hidden_dim=data["hyperparams"]["hidden_dim"],
            n_layers=data["hyperparams"]["n_layers"],
            dropout=data["hyperparams"]["dropout"],
        )
        model._model.load_state_dict(data["model_state"])
        model._model.to(model._device)
        model.is_fitted = True
        return model

    # ── Internal helpers ──────────────────────────────────────────────────────

    def _eval_loss(self, sequences, labels, criterion) -> float:
        import torch

        self._model.eval()
        total_loss = 0.0
        with torch.no_grad():
            for seq, label in zip(sequences, labels):
                x = torch.tensor(seq, dtype=torch.float32, device=self._device).unsqueeze(0)
                y = torch.tensor([[label]], dtype=torch.float32, device=self._device)
                out, _ = self._model(x)
                logit  = _sigmoid_to_logit(out[0, -1, 0].unsqueeze(0).unsqueeze(0))
                total_loss += criterion(logit, y).item()
        self._model.train()
        return total_loss / max(len(sequences), 1)

    def _check_fitted(self) -> None:
        if not self.is_fitted or self._model is None:
            raise RuntimeError("TemporalTiltModel is not fitted. Call .train() first.")


def _sigmoid_to_logit(p: "torch.Tensor") -> "torch.Tensor":
    """Invert sigmoid — used to pass probabilities into BCEWithLogitsLoss."""
    import torch
    p = torch.clamp(p, 1e-6, 1 - 1e-6)
    return torch.log(p / (1 - p))
