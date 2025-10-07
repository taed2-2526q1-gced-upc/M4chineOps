import torch
from modeling_finetune import vit_base_patch16_224


model = vit_base_patch16_224(num_classes=2, drop_path_rate=0.1)
ckpt_path = "checkpoints/checkpoint.pth"
ckpt = torch.load(ckpt_path, map_location="cpu")

# Extraer state_dict
if "model" in ckpt:
    state_dict = ckpt["model"]
elif "state_dict" in ckpt:
    state_dict = ckpt["state_dict"]
elif "module" in ckpt:
    state_dict = ckpt["module"]
else:
    state_dict = ckpt

# ❌ Eliminar la capa final (head) del checkpoint
for key in ["head.weight", "head.bias"]:
    if key in state_dict:
        del state_dict[key]

# Cargar pesos en el modelo
missing, unexpected = model.load_state_dict(state_dict, strict=False)
print("Missing keys:", missing)
print("Unexpected keys:", unexpected)
