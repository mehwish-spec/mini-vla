import torch
import numpy as np
import pandas as pd
from PIL import Image
import os
import json
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Action classes for robot manipulation
ACTION_CLASSES = ['PICK', 'PLACE', 'PUSH', 'OPEN', 'CLOSE']
ACTION_TO_IDX = {action: idx for idx, action in enumerate(ACTION_CLASSES)}
IDX_TO_ACTION = {idx: action for action, idx in ACTION_TO_IDX.items()}

def generate_synthetic_dataset(n_samples=5000, save_dir='data/processed'):
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('data/raw/images', exist_ok=True)

    instructions = {
        'PICK': [
            'pick up the red block',
            'grab the object on the left',
            'lift the blue cube',
            'take the ball from the table',
            'pick the green object',
        ],
        'PLACE': [
            'place the block on the shelf',
            'put the object down carefully',
            'set the cube on the table',
            'drop the item in the box',
            'place it on the right side',
        ],
        'PUSH': [
            'push the block forward',
            'slide the object to the left',
            'move the cube away',
            'push it towards the wall',
            'shove the object gently',
        ],
        'OPEN': [
            'open the drawer',
            'pull the door open',
            'open the container',
            'unlock and open the box',
            'open the cabinet door',
        ],
        'CLOSE': [
            'close the drawer',
            'shut the door',
            'close the container lid',
            'push the door closed',
            'close the cabinet',
        ]
    }

    data = []
    np.random.seed(42)

    for i in range(n_samples):
        action = ACTION_CLASSES[i % len(ACTION_CLASSES)]
        instruction = instructions[action][i % len(instructions[action])]

        # Generate synthetic image (224x224 RGB)
        img_array = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
        img = Image.fromarray(img_array)
        img_path = f'data/raw/images/img_{i:05d}.jpg'
        img.save(img_path)

        data.append({
            'image_path': img_path,
            'instruction': instruction,
            'action': action,
            'label': ACTION_TO_IDX[action]
        })

    df = pd.DataFrame(data)

    # Split train/val/test
    train_size = int(0.7 * len(df))
    val_size = int(0.15 * len(df))

    train_df = df[:train_size]
    val_df = df[train_size:train_size + val_size]
    test_df = df[train_size + val_size:]

    train_df.to_csv(f'{save_dir}/train.csv', index=False)
    val_df.to_csv(f'{save_dir}/val.csv', index=False)
    test_df.to_csv(f'{save_dir}/test.csv', index=False)

    print(f'Dataset generated!')
    print(f'Train: {len(train_df)} samples')
    print(f'Val: {len(val_df)} samples')
    print(f'Test: {len(test_df)} samples')
    print(f'Action classes: {ACTION_CLASSES}')

    return train_df, val_df, test_df

class VLADataset(Dataset):
    def __init__(self, csv_path, transform=None):
        self.df = pd.read_csv(csv_path)
        self.transform = transform or transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])
        ])

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        image = Image.open(row['image_path']).convert('RGB')
        image = self.transform(image)
        instruction = row['instruction']
        label = int(row['label'])
        return image, instruction, label

if __name__ == '__main__':
    print('Generating dataset...')
    train_df, val_df, test_df = generate_synthetic_dataset(n_samples=5000)
    print('Done!')
