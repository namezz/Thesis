import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model import ThesisModel
from model.loss import CompositeLoss
from config import MODEL_CONFIG

class Model:
    def __init__(self, local_rank):
        self.net = ThesisModel(MODEL_CONFIG['MODEL_ARCH'])
        self.name = MODEL_CONFIG['LOGNAME']
        self.device_id = local_rank

        # Phase-aware composite loss
        phase = MODEL_CONFIG.get('PHASE', 1)
        self.loss_fn = CompositeLoss(phase=phase)

        # Move everything to GPU
        self.device()

        # train
        self.optimG = AdamW(self.net.parameters(), lr=2e-4, weight_decay=1e-4)
        
        # AMP
        self.scaler = torch.amp.GradScaler('cuda')
        
        if local_rank != -1:
            self.net = DDP(self.net, device_ids=[local_rank], output_device=local_rank, 
                          find_unused_parameters=True)

    def train(self):
        self.net.train()

    def eval(self):
        self.net.eval()

    def device(self):
        dev = torch.device("cuda", self.device_id) if self.device_id != -1 else torch.device("cuda")
        self.net.to(dev)
        if hasattr(self, 'loss_fn'): self.loss_fn.to(dev)

    def load_model(self, name=None, rank=0):
        if rank <= 0 :
            if name is None:
                name = self.name
            path = f'ckpt/{name}.pkl'
            if os.path.exists(path):
                print(f"Loading model from {path}")
                state_dict = torch.load(path, map_location='cpu', weights_only=True)
                # Handle DDP 'module.' prefix mismatch
                model_dict = self.net.state_dict()
                has_module = any(k.startswith('module.') for k in model_dict.keys())
                ckpt_has_module = any(k.startswith('module.') for k in state_dict.keys())
                if has_module and not ckpt_has_module:
                    state_dict = {f'module.{k}': v for k, v in state_dict.items()}
                elif not has_module and ckpt_has_module:
                    state_dict = {k.replace('module.', '', 1): v for k, v in state_dict.items()}
                self.net.load_state_dict(state_dict, strict=False)
            else:
                print(f"No checkpoint found at {path}, starting from scratch.")
    
    def save_model(self, rank=0, suffix=''):
        if rank == 0:
            # Save without 'module.' prefix for portability
            state_dict = self.net.module.state_dict() if hasattr(self.net, 'module') else self.net.state_dict()
            torch.save(state_dict, f'ckpt/{self.name}{suffix}.pkl')

    @torch.no_grad()
    def inference(self, img0, img1, TTA=False, timestep=0.5, fast_TTA=False):
        """
        Wrapper for benchmark scripts.
        Returns tuple (pred,) for compatibility.
        """
        pred = self.net.inference(img0, img1, TTA=TTA, timestep=timestep, fast_TTA=fast_TTA)
        return (pred,)

    @torch.no_grad()
    def hr_inference(self, img0, img1, TTA=False, down_scale=1.0, timestep=0.5, fast_TTA=False):
        """
        High-resolution inference with optional downscaling for flow estimation.
        Recommended: down_scale=0.5 for 2K, down_scale=0.25 for 4K.
        """
        pred = self.net.inference(img0, img1, TTA=TTA, scale=down_scale, timestep=timestep, fast_TTA=fast_TTA)
        return (pred,)
    
    def freeze_backbone(self):
        """Freeze backbone parameters for Phase 2 fine-tuning."""
        backbone = self.net.module.backbone if hasattr(self.net, 'module') else self.net.backbone
        for param in backbone.parameters():
            param.requires_grad = False
        print("Backbone parameters frozen.")
    
    def unfreeze_backbone(self):
        """Unfreeze backbone parameters."""
        backbone = self.net.module.backbone if hasattr(self.net, 'module') else self.net.backbone
        for param in backbone.parameters():
            param.requires_grad = True
        print("Backbone parameters unfrozen.")
    
    def update(self, imgs, gt, learning_rate=0, training=True):
        for param_group in self.optimG.param_groups:
            param_group['lr'] = learning_rate
        if training:
            self.train()
            x = torch.cat(imgs, dim=1) if isinstance(imgs, list) else imgs
            img0 = x[:, :3]
            
            with torch.amp.autocast('cuda'):
                pred, flow = self.net(x)
                loss_total, loss_dict = self.loss_fn(pred, gt, flow=flow, img0=img0)
            
            self.optimG.zero_grad()
            self.scaler.scale(loss_total).backward()
            self.scaler.step(self.optimG)
            self.scaler.update()
            return pred, loss_dict
        else: 
            self.eval()
            with torch.no_grad():
                x = torch.cat(imgs, dim=1) if isinstance(imgs, list) else imgs
                pred, _ = self.net(x)
                return pred, {}
