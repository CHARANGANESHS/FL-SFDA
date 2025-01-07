# fl_main_ladd.py

import os
import torch
import random
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader, random_split
import torchvision.transforms as T


from src.dataset import GlaSClassificationDataset
from src.utils import *
from src.net import SimpleNet  
from src.federated_ladd import FLClientLADD, FLServerLADD
from src.gradcam import GradCAM, visualize_gradcam 

if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    global_model = SimpleNet(num_classes=2).to(device)  
    global_model_path = "./models/global_model.pth"
    if os.path.exists(global_model_path):
        print(f"Loading global model from {global_model_path}")
        global_model.load_state_dict(torch.load(global_model_path, map_location=device))
    else:
        print("No global model found, starting from random init.")

    server = FLServerLADD(global_net=global_model, aggregator="fedavg", device=device)

    glas_root = "./data/GlaS"
    csv_path  = os.path.join(glas_root, "Grade.csv")
    grade_dict = parse_grade_csv(csv_path)

    transform = T.Compose([
        T.Resize((224,224)),
        T.ToTensor()
    ])
    train_dir = os.path.join(glas_root, "train", "images")
    ds = GlaSClassificationDataset(train_dir, grade_dict, transform=transform)

    splits = [len(ds)//2, len(ds) - len(ds)//2]
    ds1, ds2 = random_split(ds, splits)
    loader1 = DataLoader(ds1, batch_size=8, shuffle=True)
    loader2 = DataLoader(ds2, batch_size=8, shuffle=True)

    client1_net = SimpleNet(num_classes=2).to(device)
    client2_net = SimpleNet(num_classes=2).to(device)

    """
    client 1 uses SHOT, client 2 uses SFDADE
    """
    client1 = FLClientLADD(client_id=1, net=client1_net, local_loader=loader1, sfda_mode="SHOT", device=device, lr=1e-3, epochs=2)
    client2 = FLClientLADD(client_id=2, net=client2_net, local_loader=loader2, sfda_mode="SFDADE", device=device, lr=1e-4, epochs=2, gamma=1.0)

    clients = [client1, client2]

    """
    Number of Federated Rounds
    """
    num_rounds = 3
    for rd in range(num_rounds):
        print(f"\n=== LADD Round {rd+1}/{num_rounds} ===")
        global_params = server.get_global_params()
        for c in clients:
            c.set_params(global_params)

        client_params_list = []
        for c in clients:
            print(f"Client {c.client_id} local adapt => {c.sfda_mode}")
            c.local_adapt()
            updated = c.get_params()
            local_size = len(c.local_loader.dataset)
            client_params_list.append((updated, local_size))

        new_global = server.aggregate(client_params_list)
        server.set_global_params(new_global)

    test_dir = os.path.join(glas_root, "test", "images")
    test_ds  = GlaSClassificationDataset(test_dir, grade_dict, transform=transform)
    test_loader = DataLoader(test_ds, batch_size=8, shuffle=False)

    final_model = server.global_net
    final_model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, labels, _ in test_loader:
            imgs, labels = imgs.to(device), labels.to(device)
            logits, _ = final_model(imgs)
            preds = logits.argmax(dim=1)
            correct += (preds==labels).sum().item()
            total += labels.size(0)
    print(f"[LADD-FL] Final global test accuracy: {100.0*correct/total:.2f}%")

    print("\n=== Grad-CAM Visualization for the final global model ===")
    images, labels, basenames = next(iter(test_loader))
    images, labels = images.to(device), labels.to(device)

    gradcam = GradCAM(final_model)
    logits, cams = gradcam(images)

    visualize_gradcam(final_model, test_loader, device, title_prefix="Global Model (LADD-FL)")

    torch.save(server.get_global_params(), global_model_path)
    print(f"Saved updated global model to {global_model_path}")

