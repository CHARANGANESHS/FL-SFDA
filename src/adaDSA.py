import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import optim

class AdaDSA(nn.Module):
    def __init__(self, target_model, source_model, alpha=0.5, lambda_=0.5, lr=1e-4, epochs=5):
        super().__init__()
        self.target_model = target_model
        self.source_model = source_model
        self.alpha = alpha
        self.lambda_ = lambda_
        self.epochs = epochs

        self.source_model.eval()
        params = list(self.target_model.parameters())
        self.optimizer = optim.Adam(params, lr=lr)

        self.src_bns=[]
        for m in self.source_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.src_bns.append({
                    'running_mean': m.running_mean.detach().clone(),
                    'running_var':  m.running_var.detach().clone()
                })
        self.tgt_bns=[]
        for m in self.target_model.modules():
            if isinstance(m, nn.BatchNorm2d):
                self.tgt_bns.append(m)

    def forward(self, x):
        return self.target_model(x)

    def adapt(self, loader, device):
        self.to(device)
        self.source_model.eval()
        self.target_model.train()
        for ep in range(self.epochs):
            self._update_bn_stats()
            running_loss=0.0
            total_samples=0
            for imgs, _, _ in loader:
                imgs=imgs.to(device)
                with torch.no_grad():
                    src_logits,_=self.source_model(imgs)
                    src_probs=F.softmax(src_logits,dim=1)
                tgt_logits,_=self.target_model(imgs)
                tgt_probs=F.softmax(tgt_logits,dim=1)

                blend_probs=(1-self.lambda_)*src_probs + self.lambda_*tgt_probs
                refined_labels=blend_probs.argmax(dim=1)

                logits,_=self.target_model(imgs)
                loss_ce=F.cross_entropy(logits, refined_labels)

                self.optimizer.zero_grad()
                loss_ce.backward()
                self.optimizer.step()

                running_loss+=loss_ce.item()*imgs.size(0)
                total_samples+=imgs.size(0)
            avg_loss=running_loss/(total_samples+1e-8)
            print(f"[AdaDSA] Epoch {ep+1}/{self.epochs}, Loss={avg_loss:.4f}")

    def _update_bn_stats(self):
        for (m_tgt, data_src) in zip(self.tgt_bns, self.src_bns):
            mu_t = m_tgt.running_mean.detach()
            var_t= m_tgt.running_var.detach()
            mu_s = data_src['running_mean']
            var_s= data_src['running_var']

            mu_ts= self.alpha*mu_t+(1-self.alpha)*mu_s
            var_ts=self.alpha*(var_t+(mu_t-mu_ts)**2)+(1-self.alpha)*(var_s+(mu_s-mu_ts)**2)

            m_tgt.running_mean.copy_(mu_ts)
            m_tgt.running_var.copy_(var_ts)
