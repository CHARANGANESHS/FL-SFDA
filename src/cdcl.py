import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim

class CDCL(nn.Module):
    def __init__(self, feature_extractor, source_classifier_weights, num_classes=2, temperature=0.07, lr=1e-4, epochs=5):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.register_buffer('source_weights', source_classifier_weights.clone())
        self.num_classes = num_classes
        self.temperature = temperature
        self.epochs = epochs
        self.optimizer = optim.Adam(self.feature_extractor.parameters(), lr=lr)

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        return feats

    def adapt(self, loader, device):
        self.to(device)
        for ep in range(self.epochs):
            all_feats = []
            with torch.no_grad():
                for imgs, _, _ in loader:
                    imgs = imgs.to(device)
                    feats = self.forward(imgs)
                    feats = F.normalize(feats, dim=1)
                    all_feats.append(feats)
            all_feats = torch.cat(all_feats, dim=0)
            pseudo = self._spherical_kmeans(all_feats, k=self.num_classes)

            idx_offset=0
            running_loss=0.0
            total_samples=0
            for imgs, _, _ in loader:
                bs=imgs.size(0)
                idxs=range(idx_offset, idx_offset+bs)
                pseudo_batch = pseudo[list(idxs)]
                idx_offset += bs

                imgs=imgs.to(device)
                feats=self.forward(imgs)
                logits=self._compute_logits(feats)
                loss_ce = F.cross_entropy(logits, pseudo_batch.to(device))

                self.optimizer.zero_grad()
                loss_ce.backward()
                self.optimizer.step()

                running_loss+=loss_ce.item()*bs
                total_samples+=bs
            avg_loss=running_loss/(total_samples+1e-8)
            print(f"[CDCL] Epoch {ep+1}/{self.epochs}, Loss={avg_loss:.4f}")

    def _compute_logits(self, feats):
        w_s = self.source_weights
        logits = feats @ w_s.T / self.temperature
        return logits

    def _spherical_kmeans(self, feats, k=2, iters=5):
        N,D=feats.size()
        centroids=feats[torch.randperm(N)[:k]]
        for _ in range(iters):
            sim=feats@centroids.T
            labels=sim.argmax(dim=1)
            for i in range(k):
                cfeats=feats[labels==i]
                if cfeats.size(0)>0:
                    centroids[i]=cfeats.mean(dim=0)
        return labels
