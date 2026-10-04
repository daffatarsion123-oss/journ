"""Vectorized feature augmentation and balanced sampling on the training device."""
import numpy as np
import torch
from ..utils.arrays import to_tensor


class TensorBatches:
    def __init__(self, x, y, layout, cfg, device):
        if cfg.encoder.sequence_len != 1:
            raise ValueError("tensor_batches currently supports single windows only")
        if not 0 < cfg.training.positive_fraction < 1:
            raise ValueError("positive_fraction must be between zero and one")
        self.x = to_tensor(x, device).to(dtype=torch.float32)
        self.y = torch.as_tensor(y, dtype=torch.long, device=device)
        self.classes = [torch.nonzero(self.y == c).flatten() for c in (0, 1)]
        if any(len(rows) == 0 for rows in self.classes):
            raise ValueError("Balanced training requires both binary classes")
        self.cfg = cfg
        self.steps = cfg.training.steps_per_epoch or max(1, len(y) // cfg.training.batch_size)
        self.channel_groups = self._groups(layout.channel_indices_array(), device)
        self.band_groups = self._groups(list(layout.band_feature_indices().values()), device)

    def _groups(self, groups, device):
        mask = torch.zeros((len(groups), self.x.shape[1]), device=device)
        for row, indices in enumerate(groups):
            mask[row, torch.as_tensor(indices, device=device)] = 1
        return mask

    def _mask_groups(self, x, groups, maximum, probability):
        count = groups.shape[0]
        maximum = min(maximum, count)
        if maximum < 1 or probability == 0:
            return x
        batch = x.shape[0]
        ranks = torch.rand(batch, count, device=x.device).argsort(dim=1).argsort(dim=1)
        drops = torch.randint(1, maximum + 1, (batch, 1), device=x.device)
        enabled = torch.rand(batch, 1, device=x.device) < probability
        selected = ((ranks < drops) & enabled).float()
        return x.masked_fill((selected @ groups) > 0, 0)

    def augment(self, x):
        cfg = self.cfg.augmentation
        if not cfg.enabled:
            return x.clone()
        batch = len(x)
        enabled = torch.rand(batch, 1, device=x.device) < cfg.gaussian_noise_p
        x = x + torch.randn_like(x) * cfg.gaussian_noise_std * enabled
        enabled = torch.rand(batch, 1, device=x.device) < cfg.amplitude_scale_p
        gain = 1 + (torch.rand(batch, 1, device=x.device) * 2 - 1) * cfg.amplitude_scale_delta
        x = x * torch.where(enabled, gain, 1)
        x = self._mask_groups(x, self.channel_groups,
                              min(cfg.channel_dropout_max, max(len(self.channel_groups) - 1, 1)),
                              cfg.channel_dropout_p)
        return self._mask_groups(x, self.band_groups, cfg.freq_masking_bands_max,
                                 cfg.freq_masking_p)

    def __iter__(self):
        size = self.cfg.training.batch_size
        for _ in range(self.steps):
            if self.cfg.training.balanced_sampler:
                negative = self.classes[0][torch.randint(len(self.classes[0]), (size,), device=self.x.device)]
                positive = self.classes[1][torch.randint(len(self.classes[1]), (size,), device=self.x.device)]
                choice = torch.rand(size, device=self.x.device) < self.cfg.training.positive_fraction
                indices = torch.where(choice, positive, negative)
            else:
                indices = torch.randint(len(self.y), (size,), device=self.x.device)
            x = self.x[indices]
            views = torch.stack([self.augment(x) for _ in range(self.cfg.augmentation.n_views)], dim=1)
            yield views, self.y[indices], None

    def __len__(self):
        return self.steps
