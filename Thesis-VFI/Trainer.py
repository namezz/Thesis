import os
import torch
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim import AdamW
from model import ThesisModel
from model.loss import CompositeLoss
from config import MODEL_CONFIG

class Model:
    def __init__(self, local_rank, backbone_lr_scale=1.0):
        self.net = ThesisModel(MODEL_CONFIG['MODEL_ARCH'])
        self.name = MODEL_CONFIG['LOGNAME']
        self.device_id = local_rank
        self.backbone_lr_scale = backbone_lr_scale

        # Phase-aware composite loss
        phase = MODEL_CONFIG.get('PHASE', 1)
        self.loss_fn = CompositeLoss(phase=phase)

        # Move everything to GPU
        self.device()

        # Discriminative LR: backbone gets scaled-down LR to protect pre-trained weights
        backbone_params = list(self.net.backbone.parameters())
        backbone_ids = set(id(p) for p in backbone_params)
        other_params = [p for p in self.net.parameters() if id(p) not in backbone_ids]
        # Include adaptive loss weight parameters (log_var_*)
        loss_params = list(self.loss_fn.parameters())
        
        self.optimG = AdamW([
            {'params': backbone_params, 'lr': 2e-4 * backbone_lr_scale, 'name': 'backbone'},
            {'params': other_params + loss_params, 'lr': 2e-4, 'name': 'other'},
        ], weight_decay=1e-4)
        
        # AMP: BF16 has same exponent range as FP32, no overflow risk
        self.use_bf16 = torch.cuda.is_bf16_supported()
        if self.use_bf16:
            self.scaler = None
        else:
            self.scaler = torch.amp.GradScaler('cuda', init_scale=1024)
        
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

    def load_model(self, name=None):
        if name is None:
            name = self.name
        path = f'ckpt/{name}.pkl'
        if os.path.exists(path):
            print(f"Loading model from {path}")
            state_dict = torch.load(path, map_location='cpu', weights_only=False)
            model_dict = self.net.state_dict()
            
            # Handle DDP prefix and shape matching
            compatible = {}
            for k, v in state_dict.items():
                fixed_k = k.replace('module.', '', 1) if not any(rk.startswith('module.') for rk in model_dict.keys()) else (f'module.{k}' if not k.startswith('module.') else k)
                if fixed_k in model_dict and model_dict[fixed_k].shape == v.shape:
                    compatible[fixed_k] = v
            
            self.net.load_state_dict(compatible, strict=False)
            
            # Load optimizer state
            optim_path = f'ckpt/{name}_optim.pkl'
            if os.path.exists(optim_path):
                optim_state = torch.load(optim_path, map_location='cpu', weights_only=False)
                try:
                    if 'optimizer' in optim_state:
                        self.optimG.load_state_dict(optim_state['optimizer'])
                    print(f"  Loaded optimizer state from {optim_path}")
                except Exception as e:
                    print(f"  Warning: Could not load optimizer state: {e}")
                
                return optim_state.get('train_state', None)
        else:
            print(f"No checkpoint found at {path}")
        return None
    
    def save_model(self, rank=0, suffix='', train_state=None):
        if rank == 0:
            # Save without 'module.' prefix for portability
            state_dict = self.net.module.state_dict() if hasattr(self.net, 'module') else self.net.state_dict()
            torch.save(state_dict, f'ckpt/{self.name}{suffix}.pkl')
            # Save optimizer + training state for proper resume
            optim_dict = {
                'optimizer': self.optimG.state_dict(),
            }
            if self.scaler is not None:
                optim_dict['scaler'] = self.scaler.state_dict()
            if train_state is not None:
                optim_dict['train_state'] = train_state
            torch.save(optim_dict, f'ckpt/{self.name}{suffix}_optim.pkl')

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
    
    def update(self, imgs, gt, learning_rate=0, training=True, accumulate=False):
        # Discriminative LR: backbone gets scaled LR
        for param_group in self.optimG.param_groups:
            if param_group.get('name') == 'backbone':
                param_group['lr'] = learning_rate * self.backbone_lr_scale
            else:
                param_group['lr'] = learning_rate
        if training:
            self.train()
            x = torch.cat(imgs, dim=1) if isinstance(imgs, list) else imgs
            img0 = x[:, :3]
            
            amp_dtype = torch.bfloat16 if self.use_bf16 else torch.float16
            with torch.amp.autocast('cuda', dtype=amp_dtype):
                pred, flow = self.net(x)
                # pred is now a list of multi-scale predictions [finest, ..., coarsest]
                # CompositeLoss handles list pred when multiscale_weights is set
                # Split combined flow into forward/backward for occlusion-aware loss
                flow_bwd = None
                if flow is not None and flow.shape[1] >= 4:
                    flow_bwd = flow[:, 2:4]
                loss_total, loss_dict = self.loss_fn(
                    pred, gt, flow=flow, flow_backward=flow_bwd, img0=img0
                )
            
            if not accumulate:
                self.optimG.zero_grad()
            if self.scaler is not None:
                self.scaler.scale(loss_total).backward()
            else:
                loss_total.backward()
            if not accumulate:
                if self.scaler is not None:
                    self.scaler.unscale_(self.optimG)
                torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
                if self.scaler is not None:
                    self.scaler.step(self.optimG)
                    self.scaler.update()
                else:
                    self.optimG.step()
            # Return finest prediction for visualization/metrics
            pred_out = pred[0] if isinstance(pred, (list, tuple)) else pred
            return pred_out, loss_dict
        else: 
            self.eval()
            with torch.no_grad():
                x = torch.cat(imgs, dim=1) if isinstance(imgs, list) else imgs
                pred, _ = self.net(x)
                pred_out = pred[0] if isinstance(pred, (list, tuple)) else pred
                return pred_out, {}

    def accum_step(self):
        """Perform optimizer step after gradient accumulation."""
        if self.scaler is not None:
            self.scaler.unscale_(self.optimG)
        torch.nn.utils.clip_grad_norm_(self.net.parameters(), max_norm=1.0)
        if self.scaler is not None:
            self.scaler.step(self.optimG)
            self.scaler.update()
        else:
            self.optimG.step()
        self.optimG.zero_grad()
