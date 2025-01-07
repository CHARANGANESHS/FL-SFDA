import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import numpy as np

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


def visualize_gradcam(model, loader, device, title_prefix="Model"):
    gradcam=GradCAM(model)
    model.eval()
    images,labels,_=next(iter(loader))
    images=images.to(device)
    labels=labels.to(device)

    with torch.no_grad():
        logits,_=model(images)
        preds=logits.argmax(dim=1)

    logits,cams=gradcam(images)
    cam0=cams[0].cpu().detach().numpy()

    import torch.nn.functional as F2
    cam0_up=F2.interpolate(
        torch.tensor(cam0).unsqueeze(0).unsqueeze(0),
        size=(224,224),mode='bilinear',align_corners=False
    ).squeeze().numpy()

    img0=images[0].cpu().detach().numpy().transpose(1,2,0)
    img0=np.clip(img0,0,1)

    plt.figure(figsize=(10,4))
    plt.suptitle(f"{title_prefix} Grad-CAM\nLabel={labels[0].item()}, Pred={preds[0].item()}")
    plt.subplot(1,3,1)
    plt.title("Original")
    plt.imshow(img0)
    plt.axis('off')

    plt.subplot(1,3,2)
    plt.title("CAM Heatmap")
    plt.imshow(cam0_up,cmap='jet')
    plt.axis('off')

    plt.subplot(1,3,3)
    plt.title("Overlay")
    plt.imshow(img0,alpha=0.6)
    plt.imshow(cam0_up,cmap='jet',alpha=0.4)
    plt.axis('off')
    plt.show()

