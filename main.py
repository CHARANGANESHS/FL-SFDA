import os
import torch
import torchvision
import torch.nn as nn
from src.adaDSA import AdaDSA
from src.gradcam import GradCAM
from src.shot import SHOT
from src.cdcl import CDCL
from src.sfdade import SFDADE
from src.net import SimpleNet
from src.dataset import GlaSClassificationDataset
from src.utils import *
from torch.utils.data import DataLoader
import numpy as np
import torch.optim as optim
import torchvision.transforms as T
import matplotlib.pyplot as plt


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


if __name__ == "__main__":
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Directory structure
    glas_root="./data/GlaS"
    csv_path =os.path.join(glas_root,"Grade.csv")
    train_dir=os.path.join(glas_root,"train","images")
    test_dir =os.path.join(glas_root,"test","images")

    # Parse CSV
    grade_dict=parse_grade_csv(csv_path)

    # Datasets
    transform=T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])
    train_ds=GlaSClassificationDataset(train_dir,grade_dict,transform=transform)
    test_ds =GlaSClassificationDataset(test_dir, grade_dict,transform=transform)
    train_loader=DataLoader(train_ds,batch_size=8,shuffle=True)
    test_loader =DataLoader(test_ds,batch_size=8,shuffle=False)

    # Source model
    src_path="source_model.pth"
    source_model=SimpleNet(num_classes=2).to(device)
    if not os.path.exists(src_path):
        print("Training source model from scratch on GlaS train set...")
        opt=optim.Adam(source_model.parameters(),lr=1e-3)
        crit=nn.CrossEntropyLoss()
        for ep in range(5):
            correct,total,running_loss=0,0,0.0
            source_model.train()
            for imgs,labels,_ in train_loader:
                imgs,labels=imgs.to(device),labels.to(device)
                opt.zero_grad()
                logits,_=source_model(imgs)
                loss=crit(logits,labels)
                loss.backward()
                opt.step()
                running_loss+=loss.item()*imgs.size(0)
                correct+=(logits.argmax(dim=1)==labels).sum().item()
                total+=imgs.size(0)
            print(f"Ep {ep+1}/5, Loss={running_loss/total:.4f}, Acc={100.0*correct/total:.2f}%")
        torch.save(source_model.state_dict(),src_path)
    else:
        source_model.load_state_dict(torch.load(src_path))

    # Evaluate source
    source_model.eval()
    correct,total=0,0
    with torch.no_grad():
        for imgs,labels,_ in test_loader:
            imgs,labels=imgs.to(device),labels.to(device)
            logits,_=source_model(imgs)
            preds=logits.argmax(dim=1)
            correct+=(preds==labels).sum().item()
            total+=labels.size(0)
    print(f"[Source] Test Accuracy: {100.0*correct/total:.2f}%")

    # Helper: freeze classifier
    def freeze_classifier(net):
        for p in net.classifier.parameters():
            p.requires_grad=False

    # =============== SFDA-DE =================
    tgt_sfdade=SimpleNet(num_classes=2).to(device)
    tgt_sfdade.load_state_dict(source_model.state_dict())
    freeze_classifier(tgt_sfdade)
    sfdade=SFDADE(tgt_sfdade.features,tgt_sfdade.classifier, num_classes=2,gamma=1.0,lr=1e-4,epochs=100)

    # =============== SHOT ====================
    tgt_shot=SimpleNet(num_classes=2).to(device)
    tgt_shot.load_state_dict(source_model.state_dict())
    freeze_classifier(tgt_shot)
    shot=SHOT(tgt_shot.features,tgt_shot.classifier,num_classes=2,lr=1e-3,momentum=0.9,epochs=100)

    # =============== CDCL ====================
    tgt_cdcl=SimpleNet(num_classes=2).to(device)
    tgt_cdcl.load_state_dict(source_model.state_dict())
    freeze_classifier(tgt_cdcl)
    with torch.no_grad():
        src_weights=source_model.classifier.weight.clone()
    cdcl=CDCL(tgt_cdcl.features,src_weights,num_classes=2,temperature=0.07,lr=1e-4,epochs=100)

    # =============== AdaDSA ==================
    tgt_adadsa=SimpleNet(num_classes=2).to(device)
    tgt_adadsa.load_state_dict(source_model.state_dict())
    adadsa=AdaDSA(tgt_adadsa, source_model, alpha=0.5,lambda_=0.5,lr=1e-4,epochs=100)

    # Adapt each method
    print("\n=== SFDA-DE ===")
    sfdade.adapt(test_loader,device)

    print("\n=== SHOT ===")
    shot.adapt(test_loader,device)

    print("\n=== CDCL ===")
    cdcl.adapt(test_loader,device)

    print("\n=== AdaDSA ===")
    adadsa.adapt(test_loader,device)

    # Evaluate
    def evaluate_model(model,loader,name):
        model.eval()
        c,t=0,0
        with torch.no_grad():
            for imgs,labels,_ in loader:
                imgs,labels=imgs.to(device),labels.to(device)
                logits,_=model(imgs)
                preds=logits.argmax(dim=1)
                c+=(preds==labels).sum().item()
                t+=labels.size(0)
        acc=100.0*c/t if t>0 else 0
        print(f"[{name}] Test Accuracy: {acc:.2f}%")

    print("\n=== Final Evaluation ===")
    evaluate_model(tgt_sfdade, test_loader, "SFDA-DE")

    # For SHOT
    evaluate_model(tgt_shot, test_loader, "SHOT")

    # For CDCL, we do a custom forward:
    def cdcl_forward(model, images):
        feats=model.forward(images)
        w_s=model.source_weights
        return feats@w_s.T / model.temperature

    c,t=0,0
    tgt_cdcl.eval()
    with torch.no_grad():
        for imgs,labels,_ in test_loader:
            imgs,labels=imgs.to(device),labels.to(device)
            logits=cdcl_forward(cdcl,imgs)
            preds=logits.argmax(dim=1)
            c+=(preds==labels).sum().item()
            t+=labels.size(0)
    print(f"[CDCL] Test Accuracy: {100.0*c/t:.2f}%")

    # For AdaDSA
    evaluate_model(tgt_adadsa, test_loader, "AdaDSA")

    # Grad-CAM Visualizations
    print("\n=== Grad-CAM Visualization ===")
    def do_cam(model, loader, title):
        print(title)
        visualize_gradcam(model, loader, device, title_prefix=title)

    do_cam(source_model, test_loader, "Source Model")
    do_cam(tgt_sfdade, test_loader, "SFDA-DE")
    do_cam(tgt_shot,   test_loader, "SHOT")
    do_cam(tgt_cdcl,   test_loader, "CDCL")
    do_cam(tgt_adadsa, test_loader, "AdaDSA")
