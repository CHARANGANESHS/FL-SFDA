import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from torch import optim

class SFDADE(nn.Module):
    """
    1) PASS A (Offline): gather features for entire dataset, do spherical k-means => means,covs
    2) PASS B (Online): For each mini-batch, compute MMD-like discrepancy => backprop.
    """
    def __init__(self, feature_extractor, classifier, num_classes=2, gamma=1.0, lr=1e-4, epochs=5):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.num_classes = num_classes
        self.gamma = gamma
        self.epochs = epochs

        params = list(self.feature_extractor.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.means = None  # will store offline
        self.covs  = None

    def forward(self, x):
        feats = self.feature_extractor(x)
        if feats.ndim > 2:
            feats = feats.view(feats.size(0), -1)
        logits = self.classifier(feats)
        return logits, feats

    def adapt(self, loader, device):
        self.to(device)
        for ep in range(self.epochs):
            # PASS A: offline
            self._build_stats(loader, device)

            # PASS B: online
            train_loss = self._pass_b(loader, device)

            print(f"[SFDA-DE] Epoch {ep+1}/{self.epochs}, Loss={train_loss:.4f}")

    def _build_stats(self, loader, device):
        """
        1) Collect features (no grad)
        2) Spherical k-means => pseudo-labels
        3) Estimate means/covs => store in self.means, self.covs
        """
        self.eval()
        all_feats = []
        with torch.no_grad():
            for imgs, _, _ in loader:
                imgs = imgs.to(device)
                # Just get the features from feature_extractor
                feats = self.feature_extractor(imgs)
                if feats.ndim > 2:
                    feats = feats.view(feats.size(0), -1)
                feats = F.normalize(feats, dim=1)
                all_feats.append(feats)
        all_feats = torch.cat(all_feats, dim=0)  # shape: N x D

        # spherical k-means
        pseudo = self._spherical_kmeans(all_feats, self.num_classes)
        # compute means, covs
        means, covs = [], []
        for k in range(self.num_classes):
            subset = all_feats[pseudo == k]
            if subset.size(0) == 0:
                d = all_feats.size(1)
                mu = torch.zeros(d, device=device)
                cv = torch.eye(d, device=device)
            else:
                mu = subset.mean(dim=0)
                diff = subset - mu
                cv = (diff.T @ diff)/ subset.size(0)
                cv = cv + self.gamma*torch.eye(diff.size(1), device=device)
            means.append(mu)
            covs.append(cv)
        self.means = torch.stack(means, dim=0)
        self.covs  = torch.stack(covs,  dim=0)

    def _pass_b(self, loader, device):
        """
        For each batch:
         - Forward pass with grad
         - For each sample, measure MMD to the distribution => sum up
         - Backprop
        """
        self.train()
        running_loss = 0.0
        total_samples = 0
        for imgs, _, _ in loader:
            imgs = imgs.to(device)
            logits, feats = self.forward(imgs)
            feats_norm = F.normalize(feats, dim=1)

            with torch.no_grad():
                sim = feats_norm @ F.normalize(self.means, dim=1).T  # B x K
                pseudo = sim.argmax(dim=1)

            cdd_loss = torch.tensor(0.0, device=device, requires_grad=True)
            for i in range(imgs.size(0)):
                c = pseudo[i]
                diff_mean = (feats[i] - self.means[c]).pow(2).sum()
                diff_cov  = self.covs[c].trace()  # placeholder
                cdd_loss  = cdd_loss + (diff_mean + diff_cov)
            cdd_loss = cdd_loss / imgs.size(0)

            self.optimizer.zero_grad()
            cdd_loss.backward()
            self.optimizer.step()

            running_loss  += cdd_loss.item() * imgs.size(0)
            total_samples += imgs.size(0)

        return running_loss / (total_samples + 1e-8)

    def _spherical_kmeans(self, feats, k=2, iters=5):
        """
        feats: (N, D)
        returns labels: (N,)
        """
        centroids = feats[torch.randperm(len(feats))[:k]].clone()
        for _ in range(iters):
            sim = feats @ centroids.T
            labels = sim.argmax(dim=1)
            for i in range(k):
                idxs = (labels == i)
                if idxs.sum() > 0:
                    centroids[i] = feats[idxs].mean(dim=0)
        return labels