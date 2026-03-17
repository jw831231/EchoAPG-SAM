import torch
from peft import LoraConfig, get_peft_model
from segment_anything import sam_model_registry
from models.adapters import MSPAd

class EnhancedSAM(torch.nn.Module):
    def __init__(self, model_type="vit_b", checkpoint=None, lora_r=16):
        super().__init__()
        sam = sam_model_registry[model_type](checkpoint=checkpoint)
        
        # 1. LoRA（只微调qkv）
        lora_config = LoraConfig(
            r=lora_r, lora_alpha=2*lora_r,
            target_modules=["qkv"], lora_dropout=0.05, bias="none"
        )
        sam.image_encoder = get_peft_model(sam.image_encoder, lora_config)
        
        # 2. MSPAd
        self.mspad_modules = {}
        for i, block in enumerate(sam.image_encoder.blocks):
            adapter = MSPAd(dim=block.norm1.normalized_shape[0]).to(sam.device)
            hook = block.register_forward_hook(self._make_mspad_hook(adapter))
            self.mspad_modules[i] = (adapter, hook)
        
        self.sam = sam
        self.image_encoder = sam.image_encoder

    def _make_mspad_hook(self, adapter):
        def hook(module, inp, out): 
            x_for_mspad = out.permute(0, 3, 1, 2)
            mspad_out = adapter(x_for_mspad)
            return out + mspad_out.permute(0, 2, 3, 1)
        return hook

    def forward(self, x):
        return self.sam.image_encoder(x)

    def save_clean_state(self, path):
        state = {k: v for k, v in self.sam.state_dict().items() 
                 if "lora" in k or "mspad" in k.lower()}
        torch.save(state, path)
