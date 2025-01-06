import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class SHOT(nn.Module):
    def __init__(self, feature_extractor, classifier, num_classes=2, lr=1e-3, momentum=0.9, epochs=5):
        super().__init__()
        self.feature_extractor = feature_extractor
        self.classifier = classifier
        self.num_classes = num_classes
        self.epochs = epochs
        params = list(self.feature_extractor.parameters()) + list(self.classifier.parameters())
        self.optimizer = optim.SGD(params, lr=lr, momentum=momentum)

    def forward(self, x):
        feats = self.feature_extractor(x)
        feats = feats.view(feats.size(0), -1)
        logits = self.classifier(feats)
        return logits, feats

    def adapt(self, loader, device):
        self.to(device)
        for ep in range(self.epochs):
            with torch.no_grad():
                sum_probs = torch.zeros(self.num_classes, device=device)
                total_count = 0
                for imgs, _, _ in loader:
                    imgs = imgs.to(device)
                    logits, _ = self.forward(imgs)
                    probs = F.softmax(logits, dim=1)
                    sum_probs += probs.sum(dim=0)
                    total_count += probs.size(0)
            p_hat = sum_probs / (total_count+1e-8)
            p_hat = torch.clamp(p_hat, min=1e-8, max=1.0)

            running_loss = 0.0
            total_samples= 0
            self.train()
            for imgs, _, _ in loader:
                imgs = imgs.to(device)
                logits, _ = self.forward(imgs)
                probs = F.softmax(logits, dim=1)
                log_probs = F.log_softmax(logits, dim=1)
                ent_loss = - (probs*log_probs).sum(dim=1).mean()
                div_loss = (p_hat*torch.log(p_hat)).sum()
                total_loss= ent_loss + div_loss

                self.optimizer.zero_grad()
                total_loss.backward()
                self.optimizer.step()

                running_loss += total_loss.item() * imgs.size(0)
                total_samples += imgs.size(0)

            avg_loss = running_loss/(total_samples+1e-8)
            print(f"[SHOT] Epoch {ep+1}/{self.epochs}, Loss={avg_loss:.4f}")
