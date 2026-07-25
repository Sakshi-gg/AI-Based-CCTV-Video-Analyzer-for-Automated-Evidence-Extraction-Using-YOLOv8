"""
anomaly_detector.py
Temporal anomaly detection using optical flow + LSTM Autoencoder.
Assigns an anomaly score (0.0 - 1.0) to each frame.
High scores indicate abnormal motion: running, fighting, loitering.

Reference: Sultani et al., CVPR 2018 - Real-world Anomaly Detection
           in Surveillance Videos.
"""

import cv2
import numpy as np
import torch
import torch.nn as nn


# ── Model Definition ──────────────────────────────────────────────────────────

class LSTMAutoencoder(nn.Module):
    """
    LSTM Autoencoder for temporal anomaly detection.
    Input:  sequence of optical flow feature vectors
    Output: reconstructed sequence
    Anomaly score = mean reconstruction error
    """
    def __init__(self, input_size=256, hidden_size=128, num_layers=2):
        super().__init__()
        self.encoder = nn.LSTM(
            input_size=input_size,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True
        )
        self.decoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=input_size,
            num_layers=num_layers,
            batch_first=True
        )

    def forward(self, x):
        # Encode
        _, (hidden, cell) = self.encoder(x)
        # Repeat hidden state as decoder input
        decoder_input = hidden[-1].unsqueeze(1).repeat(1, x.size(1), 1)
        # Decode
        output, _ = self.decoder(decoder_input)
        return output


# ── Feature Extraction ────────────────────────────────────────────────────────

def extract_optical_flow_features(prev_gray, curr_gray, feature_size=256):
    """
    Computes dense optical flow between two grayscale frames.
    Returns a flattened, normalized feature vector of fixed size.
    """
    flow = cv2.calcOpticalFlowFarneback(
        prev_gray, curr_gray,
        None,
        pyr_scale=0.5,
        levels=3,
        winsize=15,
        iterations=3,
        poly_n=5,
        poly_sigma=1.2,
        flags=0
    )

    # Magnitude of flow vectors
    magnitude, _ = cv2.cartToPolar(flow[..., 0], flow[..., 1])

    # Resize to fixed spatial grid and flatten
    resized = cv2.resize(magnitude, (16, 16))
    flattened = resized.flatten()  # 256 features

    # Normalize
    max_val = flattened.max()
    if max_val > 0:
        flattened = flattened / max_val

    return flattened.astype(np.float32)


# ── Anomaly Detector Class ────────────────────────────────────────────────────

class AnomalyDetector:
    """
    Wraps the LSTM Autoencoder with a sliding window buffer.
    Call update(frame) on each frame — returns anomaly score when
    the buffer has enough frames, otherwise returns 0.0.
    """

    SEQUENCE_LEN  = 16   # frames per sequence
    FEATURE_SIZE  = 256  # 16x16 optical flow grid
    HIDDEN_SIZE   = 128
    NUM_LAYERS    = 2
    # Score above this threshold → flagged as anomalous
    ANOMALY_THRESHOLD = 0.45

    def __init__(self, device=None):
        self.device = device or (
            torch.device('cuda') if torch.cuda.is_available()
            else torch.device('cpu')
        )
        self.model = LSTMAutoencoder(
            input_size=self.FEATURE_SIZE,
            hidden_size=self.HIDDEN_SIZE,
            num_layers=self.NUM_LAYERS
        ).to(self.device)
        self.model.eval()

        # Sliding window buffer of flow feature vectors
        self.feature_buffer = []
        self.prev_gray      = None

        # Running stats for adaptive normalization
        self.score_history  = []

    def reset(self):
        """Call this at the start of each new video analysis."""
        self.feature_buffer = []
        self.prev_gray      = None
        self.score_history  = []

    def update(self, frame_bgr):
        """
        Process one frame. Returns (anomaly_score, is_anomalous).
        anomaly_score: float 0.0 - 1.0
        is_anomalous:  bool  (score > ANOMALY_THRESHOLD)
        """
        curr_gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        curr_gray = cv2.resize(curr_gray, (320, 240))

        if self.prev_gray is None:
            self.prev_gray = curr_gray
            return 0.0, False

        # Extract optical flow features
        features = extract_optical_flow_features(
            self.prev_gray, curr_gray, self.FEATURE_SIZE
        )
        self.prev_gray = curr_gray
        self.feature_buffer.append(features)

        # Need full sequence to score
        if len(self.feature_buffer) < self.SEQUENCE_LEN:
            return 0.0, False

        # Keep only last SEQUENCE_LEN frames
        if len(self.feature_buffer) > self.SEQUENCE_LEN:
            self.feature_buffer.pop(0)

        # Build tensor: (1, seq_len, feature_size)
        sequence = np.stack(self.feature_buffer, axis=0)
        tensor   = torch.tensor(sequence, dtype=torch.float32).unsqueeze(0).to(self.device)

        # Forward pass — reconstruction
        with torch.no_grad():
            reconstructed = self.model(tensor)

        # Reconstruction error = anomaly score
        error = torch.mean((tensor - reconstructed) ** 2).item()

        # Normalize score to 0-1 using running history
        self.score_history.append(error)
        if len(self.score_history) > 100:
            self.score_history.pop(0)

        min_e = min(self.score_history)
        max_e = max(self.score_history)

        if max_e - min_e > 1e-6:
            score = (error - min_e) / (max_e - min_e)
        else:
            score = 0.0

        score = float(np.clip(score, 0.0, 1.0))
        return score, score > self.ANOMALY_THRESHOLD
