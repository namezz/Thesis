import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model import ThesisModel
from model.loss import LapLoss, VGGPerceptualLoss
from config import MODEL_CONFIG

class Model:
    def __init__(self, local_rank):
        self.net = ThesisModel(MODEL_CONFIG['MODEL_ARCH'])
        self.name = MODEL_CONFIG['LOGNAME']
        self.device_id = local_rank
        self.device()

        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        self.lap_loss = LapLoss()
        self.vgg_loss = VGGPerceptualLoss()
        
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        dev = torch.device("cuda", self.device_id) if self.device_id != -1 else torch.device("cuda")
        self.net.to(dev)
        if hasattr(self, 'lap_loss'): self.lap_loss.to(dev)
        if hasattr(self, 'vgg_loss'): self.vgg_loss.to(dev)

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
    def inference(self, img0, img1, TTA=False, timestep=0.5, fast_TTA=False):
        # Wrapper for benchmark scripts
        # TODO: Implement TTA logic if needed
        return self.net.inference(img0, img1, timestep=timestep)

    @torch.no_grad()
    def hr_inference(self, img0, img1, TTA=False, down_scale=1.0, timestep=0.5, fast_TTA=False):
        # Placeholder for high-res inference logic
        return self.inference(img0, img1, TTA=TTA, timestep=timestep, fast_TTA=fast_TTA)
    
    def update(self, imgs, gt, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
            pred = self.net(imgs)
            
            # Hybrid Loss
            loss_lap = self.lap_loss(pred, gt)
            loss_vgg = self.vgg_loss(pred, gt)
            loss_total = loss_lap + 0.01 * loss_vgg
            
            self.optimG.zero_grad()
            loss_total.backward()
            self.optimG.step()
            return pred, loss_total
        else: 
            self.eval()
            with torch.no_grad():
                pred = self.net(imgs)
                return pred, 0
