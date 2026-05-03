import torch
import torch.nn as nn
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import io
import base64
from torchvision import transforms
from app.model import VLAModel, ACTION_CLASSES

def get_attention_map(model, image, instruction):
    attention_weights = []

    def hook_fn(module, input, output):
        if hasattr(output, 'attentions') and output.attentions is not None:
            attention_weights.append(output.attentions)

    model.eval()
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
    ])

    if not isinstance(image, torch.Tensor):
        img_tensor = transform(image)
    else:
        img_tensor = image

    with torch.no_grad():
        inputs = model.clip.processor if hasattr(model.clip, 'processor') else None
        processed = model.processor(
            text=[instruction],
            images=[img_tensor],
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77
        )
        outputs = model.clip(**processed, output_attentions=True)

    # Get vision attention from last layer
    if outputs.vision_model_output.attentions:
        last_attn = outputs.vision_model_output.attentions[-1]
        attn_map = last_attn[0].mean(0)[0, 1:]  # CLS token attention
        grid_size = int(attn_map.shape[0] ** 0.5)
        attn_map = attn_map.reshape(grid_size, grid_size).numpy()
        attn_map = (attn_map - attn_map.min()) / (attn_map.max() - attn_map.min() + 1e-8)
        return attn_map
    return None

def overlay_attention(image, attn_map, alpha=0.5):
    if isinstance(image, torch.Tensor):
        img_np = image.permute(1, 2, 0).numpy()
        img_np = (img_np * 255).astype(np.uint8)
        image = Image.fromarray(img_np)

    img_resized = image.resize((224, 224))
    img_np = np.array(img_resized)

    attn_resized = np.array(
        Image.fromarray((attn_map * 255).astype(np.uint8)).resize((224, 224))
    ) / 255.0

    heatmap = cm.jet(attn_resized)[:, :, :3]
    heatmap = (heatmap * 255).astype(np.uint8)

    overlay = (alpha * heatmap + (1 - alpha) * img_np).astype(np.uint8)
    return Image.fromarray(overlay)

def image_to_base64(image):
    buffer = io.BytesIO()
    image.save(buffer, format='PNG')
    return base64.b64encode(buffer.getvalue()).decode()

if __name__ == '__main__':
    print('Testing attention visualization...')
    model = VLAModel(num_classes=5)
    import os
    if os.path.exists('models/checkpoints/best_model.pt'):
        model.load_state_dict(torch.load('models/checkpoints/best_model.pt', map_location='cpu'))
    model.eval()

    img = Image.open('data/raw/images/img_00001.jpg').convert('RGB')
    attn_map = get_attention_map(model, img, 'pick up the red block')

    if attn_map is not None:
        overlay = overlay_attention(img, attn_map)
        overlay.save('data/attention_test.png')
        print('Attention map saved to data/attention_test.png')
    else:
        print('Attention visualization not available')
    print('Done!')
