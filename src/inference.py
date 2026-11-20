"""Slice-by-slice prediction of a whole 3D volume."""
import numpy as np
import torch

from .dataset import window_normalise


@torch.no_grad()
def predict_volume(model, vol_hu, device, hu_window, batch_size=16, amp=True, tta_flip=False):
    """Predict foreground probability for every axial slice; returns float32 array (z, y, x)."""
    model.eval()
    use_amp = amp and device.type == "cuda"
    out = np.zeros(vol_hu.shape, dtype=np.float32)
    for i in range(0, vol_hu.shape[0], batch_size):
        chunk = np.asarray(vol_hu[i:i + batch_size])
        x = torch.from_numpy(window_normalise(chunk, *hu_window)).unsqueeze(1).to(device)
        with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
            logits = model(x)
        prob = torch.sigmoid(logits.float())
        if tta_flip:
            with torch.autocast(device_type=device.type, dtype=torch.float16, enabled=use_amp):
                logits_f = model(x.flip(-1))
            prob = (prob + torch.sigmoid(logits_f.float()).flip(-1)) / 2.0
        out[i:i + chunk.shape[0]] = prob[:, 0].cpu().numpy()
    return out
