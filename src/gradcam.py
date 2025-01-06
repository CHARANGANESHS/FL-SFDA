import torch
import torch.nn.functional as F

class GradCAM:
    def __init__(self, model):
        self.model = model
        self.gradients = None
        self.activations = None

        def forward_hook(m, i, o):
            self.activations = o

        def backward_hook(m, gi, go):
            self.gradients = go[0]
        self.model.features.register_forward_hook(forward_hook)
        self.model.features.register_backward_hook(backward_hook)

    def __call__(self, x, class_idx=None):
        self.model.zero_grad()
        logits, feat = self.model(x)
        if class_idx is None:
            class_idx = logits.argmax(dim=1)
        one_hot = torch.zeros_like(logits)
        for i in range(logits.size(0)):
            one_hot[i, class_idx[i]] = 1.0
        logits.backward(gradient=one_hot, retain_graph=True)
        grads = self.gradients
        acts = self.activations
        alpha = grads.view(grads.size(0), grads.size(1), -1).mean(dim=2)
        cams = []
        for i in range(acts.size(0)):
            w_acts = alpha[i].unsqueeze(-1).unsqueeze(-1)*acts[i]
            cam = w_acts.sum(dim=0)
            cam = F.relu(cam)
            mi, ma = cam.min(), cam.max()
            cam = (cam-mi)/(ma-mi+1e-8)
            cams.append(cam)
        cams = torch.stack(cams, dim=0)
        return logits, cams
