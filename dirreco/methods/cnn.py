"""CNN regression of the direction (models trained with cnn/train.py).

    cnn: {type: cnn, checkpoint: cnn/checkpoints/resnet50_p50_continuous/best.pt}
    cnn: {type: cnn, checkpoint: [cnn/checkpoints/a/best.pt, cnn/checkpoints/b/best.pt]}

Each checkpoint is trained for one pitch (and representation); with several, the one whose
pitch matches the grid point is used and pitches without a model are skipped. Like the
centroid, the CNN needs no reference sample: the theta = 0 template is learned from the
training pool instead of being estimated from N reference events.

IMPORTANT: train/eval separation. The network was trained on events with index >= its
--holdout (default 2,000,000), so a study using it must set `data: {max_events: <holdout>}`.

Options: device (cuda | cpu, default cuda if available), plus the common max_N/min_pitch/max_grid.
The model is built lazily in the worker, so CUDA is never initialised before the engine forks.
"""
from pathlib import Path

import numpy as np

from . import Method, register
from ..data import REPO


@register("cnn")
class CNN(Method):
    needs_ref = False

    def __init__(self, checkpoint, device=None, **common):
        import torch
        super().__init__(**common)
        self.device = device
        self._ckpts, self._models = {}, {}
        for path in ([checkpoint] if isinstance(checkpoint, str) else list(checkpoint)):
            path = Path(path) if Path(path).is_absolute() else REPO / path
            ck = torch.load(path, map_location="cpu", weights_only=True)
            pitch = float(ck["image_spec"]["pitch"])
            if pitch in self._ckpts:
                raise ValueError(f"two CNN checkpoints for pitch {pitch:g} mm")
            self._ckpts[pitch] = ck

    def applicable(self, N, geom):
        return super().applicable(N, geom) and float(geom.pitch) in self._ckpts

    def _model(self, pitch):
        if pitch not in self._models:
            import torch
            from cnn.images import ImageSpec
            from cnn.resnet import build_model
            ck = self._ckpts[pitch]
            a = ck["args"]
            dev = torch.device(self.device or ("cuda" if torch.cuda.is_available() else "cpu"))
            model = build_model(a["arch"], num_classes=2, channels=1, norm=a["norm"], stem=a["stem"])
            model.load_state_dict(ck["model_state_dict"])
            self._models[pitch] = (model.to(dev).eval(), ImageSpec(**ck["image_spec"]), dev)
        return self._models[pitch]

    def estimate(self, meas, ref, geom):
        import torch
        from cnn.images import make_images
        model, spec, dev = self._model(float(geom.pitch))
        vtx = torch.as_tensor(meas.vtx[:, :2], dtype=torch.float64, device=dev)
        cap = torch.as_tensor(meas.cap[:, :2], dtype=torch.float64, device=dev)
        img_id = torch.zeros(len(vtx), dtype=torch.long, device=dev)
        x, _ = make_images(vtx, cap, img_id, 1, spec)
        with torch.no_grad():
            u = model(x)[0].double().cpu().numpy()
        return float(np.degrees(np.arctan2(u[1], u[0])))
