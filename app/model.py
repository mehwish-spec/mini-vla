import torch
import torch.nn as nn
from transformers import CLIPModel, CLIPProcessor
import mlflow

ACTION_CLASSES = ['PICK', 'PLACE', 'PUSH', 'OPEN', 'CLOSE']

class VLAModel(nn.Module):
    def __init__(self, num_classes=5, dropout=0.3):
        super(VLAModel, self).__init__()

        # Load CLIP model
        print('Loading CLIP...')
        self.clip = CLIPModel.from_pretrained('openai/clip-vit-base-patch32')
        self.processor = CLIPProcessor.from_pretrained('openai/clip-vit-base-patch32')

        # Freeze CLIP weights
        for param in self.clip.parameters():
            param.requires_grad = False

        # Fusion layer - combines vision + language
        clip_dim = 512
        self.fusion = nn.Sequential(
            nn.Linear(clip_dim * 2, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
        )

        # Action prediction head
        self.action_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(128, num_classes)
        )

    def forward(self, images, instructions):
        inputs = self.processor(
            text=instructions,
            images=images,
            return_tensors='pt',
            padding=True,
            truncation=True,
            max_length=77
        )
        inputs = {k: v.to(next(self.parameters()).device) for k, v in inputs.items()}

        outputs = self.clip(**inputs)
        vision_features = outputs.image_embeds
        text_features = outputs.text_embeds

        # Normalize
        vision_features = vision_features / vision_features.norm(dim=-1, keepdim=True)
        text_features = text_features / text_features.norm(dim=-1, keepdim=True)

        # Fuse vision + language
        fused = torch.cat([vision_features, text_features], dim=-1)
        fused = self.fusion(fused)

        # Predict action
        logits = self.action_head(fused)
        return logits

    def predict(self, image, instruction):
        self.eval()
        with torch.no_grad():
            logits = self.forward([image], [instruction])
            probs = torch.softmax(logits, dim=-1)
            pred_idx = probs.argmax(dim=-1).item()
            confidence = probs[0][pred_idx].item()
        return ACTION_CLASSES[pred_idx], round(confidence, 4)

if __name__ == '__main__':
    print('Testing VLA Model...')
    model = VLAModel(num_classes=5)
    print('Model created!')
    print('Parameters:')
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f'  Total: {total:,}')
    print(f'  Trainable: {trainable:,}')
    print(f'  Frozen (CLIP): {total - trainable:,}')
    print('Model test passed!')
