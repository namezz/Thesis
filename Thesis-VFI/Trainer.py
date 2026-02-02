import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model import ThesisModel
from config import MODEL_CONFIG

class Model:
    def __init__(self, local_rank):
        self.net = ThesisModel(MODEL_CONFIG['MODEL_ARCH'])
        self.name = MODEL_CONFIG['LOGNAME']
        self.device_id = local_rank
        self.net.to(torch.device("cuda", local_rank) if local_rank != -1 else "cuda")

        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def load_model(self, name=None, rank=0):
        if rank <= 0 :
            if name is None:
                name = self.name
            path = f'ckpt/{name}.pkl'
            if os.path.exists(path):
                print(f"Loading model from {path}")
                self.net.load_state_dict(torch.load(path), strict=False)
            else:
                print(f"No checkpoint found at {path}, starting from scratch.")
    
    def save_model(self, rank=0):
        if rank == 0:
            torch.save(self.net.state_dict(), f'ckpt/{self.name}.pkl')

    @torch.no_grad()
    def inference(self, img0, img1, TTA=False, fast_TTA=False):
        # Wrapper for benchmark scripts
        return self.net.inference(img0, img1)
    
    def update(self, imgs, gt, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
            pred = self.net(imgs)
            loss_l1 = F.l1_loss(pred, gt)
            
            self.optimG.zero_grad()
            loss_l1.backward()
            self.optimG.step()
            return pred, loss_l1
        else: 
            self.eval()
            with torch.no_grad():
                pred = self.net(imgs)
                return pred, 0
