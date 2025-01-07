# federated_ladd.py

import copy
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import random

# Import your SFDA methods from existing modules
from src.sfdade import SFDADE   # or wherever you store SFDA-DE
from src.shot import SHOT
from src.cdcl import CDCL
from src.adaDSA import AdaDSA

###############################################################################
# LADD-Style Federated Classes
###############################################################################

class FLClientLADD:
    """
    A client in the LADD (Learning Across Domains and Devices) scenario.

    Each client has:
      - local unlabeled data (for SFDA)
      - a style representation s_k (extracted or estimated)
      - a local model that can run SFDA adaptation.

    The LADD paper references eq. (1), eq. (7), eq. (8), etc.:

      (1) w^* = arg min_{w} sum_{k in K} (|D^k| / |D^T|) * L_k(w)
          -> This is analogous to FedAvg, weighting by dataset size.

      We'll also incorporate the style-based clustering step (Algorithm 2 in snippet),
      which groups clients by their style s_k for more specialized model updates.
    """
    def __init__(self, client_id, net, local_loader, sfda_mode="SHOT",
                 device="cpu", style_vector=None, **sfda_kwargs):
        """
        net: the local net initialized from server
        local_loader: local DataLoader for unlabeled data
        sfda_mode: e.g. "SFDADE", "SHOT", "CDCL", "AdaDSA"
        style_vector: s_k for style-based clustering
        sfda_kwargs: extra hyperparams (lr, epochs, gamma, etc.)
        """
        self.client_id = client_id
        self.device = device
        self.local_loader = local_loader
        self.sfda_mode = sfda_mode
        self.style_vector = style_vector  # or style code from eq. (8) or eq. (7) in snippet
        self.net = net.to(self.device)

        self.sfda_method = self._build_sfda_method(self.net, sfda_mode, **sfda_kwargs)

    def _build_sfda_method(self, net, mode, **kwargs):
        if mode == "SFDADE":
            return SFDADE(
                feature_extractor=net.features,
                classifier=net.classifier,
                num_classes=2,
                **kwargs
            )
        elif mode == "SHOT":
            return SHOT(
                feature_extractor=net.features,
                classifier=net.classifier,
                num_classes=2,
                **kwargs
            )
        elif mode == "CDCL":
            source_weights = kwargs.pop("source_classifier_weights")
            return CDCL(
                feature_extractor=net.features,
                source_classifier_weights=source_weights,
                num_classes=2,
                **kwargs
            )
        elif mode == "AdaDSA":
            source_model = kwargs.pop("source_model")
            return AdaDSA(
                target_model=net,
                source_model=source_model,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown SFDA mode: {mode}")

    def local_adapt(self):
        """
        Step 3 in LADD (Algorithm 1):
        Each client does local adaptation with chosen SFDA approach.
        """
        self.sfda_method.to(self.device)
        self.sfda_method.adapt(self.local_loader, self.device)

    def get_params(self):
        """
        Return local net's parameters after adaptation.
        """
        return self.net.state_dict()

    def set_params(self, new_params):
        """
        Overwrite local net with aggregator-provided parameters.
        """
        self.net.load_state_dict(new_params)
        # also update the submodules in sfda_method
        if hasattr(self.sfda_method, 'feature_extractor'):
            self.sfda_method.feature_extractor.load_state_dict(self.net.features.state_dict())
        if hasattr(self.sfda_method, 'classifier'):
            self.sfda_method.classifier.load_state_dict(self.net.classifier.state_dict())

class FLServerLADD:
    """
    The LADD server coordinates the FL process, referencing eq. (1), eq. (7), eq. (8), etc.
    This class:
      - Maintains a global model
      - Possibly clusters clients by style (Algorithm 2 in snippet).
      - Aggregates local updates using a FedAvg-like approach or a cluster-based approach.
    """
    def __init__(self, global_net, aggregator="fedavg", device="cpu"):
        self.global_net = global_net
        self.aggregator = aggregator
        self.device = device

    def get_global_params(self):
        return self.global_net.state_dict()

    def set_global_params(self, new_params):
        self.global_net.load_state_dict(new_params)

    def style_based_clustering(self, clients, hyperparams=None):
        """
        Implementation of snippet's Algorithm 2 (Clustering Selection).
        - Each client has style_vector s_k.
        - We run K-Means or another approach to find clusters.
        - Return a list of clusters, each cluster is a list of client indices.

        eq. (8) in the snippet references style extraction or pseudo-labeling.
        eq. (7) might refer to some style-based aggregator logic.

        We'll do a minimal approach: if style vectors exist, do a quick K-Means on them.
        """
        from sklearn.cluster import KMeans

        style_mat = []
        for c in clients:
            if c.style_vector is not None:
                style_mat.append(c.style_vector)
            else:
                # fallback: random style
                dim = 4
                style_mat.append(torch.randn(dim))
        style_mat = torch.stack(style_mat, dim=0).cpu().numpy()

        n_clusters = hyperparams.get("n_clusters", 2)
        kmeans = KMeans(n_clusters=n_clusters, random_state=42).fit(style_mat)
        labels = kmeans.labels_

        clusters = {}
        for idx, lab in enumerate(labels):
            if lab not in clusters:
                clusters[lab] = []
            clusters[lab].append(idx)  # client index
        return clusters  # dict { cluster_label -> [client indices] }

    def aggregate(self, client_params_list):
        """
        eq. (1) from LADD: w^* = arg min sum_{k in K} (|D^k|/|D^T|) L_k(w)
        Approximated by FedAvg weighting.

        client_params_list: [ (params, n_samples), ... ]
        """
        new_state = copy.deepcopy(client_params_list[0][0])
        for k in new_state.keys():
            new_state[k] = 0.0
        total_samples = 0
        for (params, n) in client_params_list:
            total_samples += n
        for (params, n) in client_params_list:
            weight = n / total_samples
            for key in new_state.keys():
                new_state[key] += weight * params[key]
        return new_state
