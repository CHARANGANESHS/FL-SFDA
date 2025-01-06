import torch.nn as nn

class SimpleNet(nn.Module):
    """
    A minimal CNN with a (features, classifier) structure.
    """
    def __init__(self, num_classes=2):
        super(SimpleNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.Conv2d(16, 32, kernel_size=3, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool2d((4,4))
        )
        self.classifier = nn.Linear(32*4*4, num_classes)

    def forward(self, x):
        feat = self.features(x)          # shape: B x 32 x 4 x 4
        flatten = feat.view(feat.size(0), -1)
        logits = self.classifier(flatten)
        return logits, feat