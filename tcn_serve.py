# tcn_serve.py (normalize with saved mu/sd; reject wrong shapes)

import numpy as np
import torch
import torch.nn as nn
from flask import Flask, request, jsonify

from tcn_train import TCN, FEATURE_COLS

app = Flask(__name__)

CKPT_PATH = "tcn_model.pt"
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def load_model():
    # Try the new default first; if it fails, allowlist numpy reconstruct OR use weights_only=False
    try:
        ckpt = torch.load(CKPT_PATH, map_location=DEVICE)  # PyTorch 2.6 default: weights_only=True
    except Exception:
        # Trust your own checkpoint? Allowlist the numpy reconstruct symbol or force weights_only=False
        try:
            import numpy as np
            torch.serialization.add_safe_globals([np.core.multiarray._reconstruct])
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE)  # try again with the allowlist
        except Exception:
            # Final fallback: explicitly disable weights_only safety (ONLY if you trust the file)
            ckpt = torch.load(CKPT_PATH, map_location=DEVICE, weights_only=False)

    model = TCN(input_size=len(FEATURE_COLS))
    model.load_state_dict(ckpt["model_state"])
    model.to(DEVICE).eval()

    seq_len = int(ckpt["seq_len"])
    # mu/sd may be saved as numpy arrays or lists; normalize to float32 numpy arrays
    mu = ckpt.get("mu", np.zeros(len(FEATURE_COLS), dtype=np.float32))
    sd = ckpt.get("sd", np.ones(len(FEATURE_COLS), dtype=np.float32))
    mu = np.asarray(mu, dtype=np.float32)
    sd = np.asarray(sd, dtype=np.float32)
    return model, seq_len, mu, sd

model, SEQ_LEN, MU, SD = load_model()

@app.route("/predict", methods=["POST"])
def predict():
    try:
        payload = request.get_json(force=True)
        seq = payload.get("sequence", None)
        if seq is None:
            return jsonify(error="Missing 'sequence'"), 400

        arr = np.array(seq, dtype=np.float32)  # [T, F]
        if arr.ndim != 2 or arr.shape[1] != len(FEATURE_COLS):
            return jsonify(error=f"sequence must be [T, {len(FEATURE_COLS)}]"), 400
        if arr.shape[0] < SEQ_LEN:
            return jsonify(error=f"T too small: got {arr.shape[0]}, need >= {SEQ_LEN}"), 400

        # keep last SEQ_LEN and normalize with train stats
        arr = arr[-SEQ_LEN:, :]
        arr = (arr - MU) / SD

        X = torch.from_numpy(arr).unsqueeze(0).to(DEVICE)  # [1, T, F]
        with torch.no_grad():
            logits = model(X)              # [1, 2]
            probs = torch.softmax(logits, dim=1).cpu().numpy()[0]

        return jsonify(p_long=float(probs[0]), p_short=float(probs[1]))
    except Exception as e:
        return jsonify(error=str(e)), 500

if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)

# python tcn_serve.py
