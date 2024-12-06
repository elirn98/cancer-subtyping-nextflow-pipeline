from utils.data_loader import OvarianDataset
import torch
import torchvision
from torchvision import transforms
from torch import nn
from networks import FeatureExtractor
from tqdm import tqdm
import shutil
import os
import argparse
import pickle

def save_checkpoint(args, state, save_name, save_dir):
    # This Function save model checkpoint in save_dir
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = save_dir + "/" + "model" + ".pth.tar"
    torch.save(state, filename)


def print_memory_usage():
    # Total memory in bytes
    total_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_PHYS_PAGES')
    # Available memory in bytes
    available_memory = os.sysconf('SC_PAGE_SIZE') * os.sysconf('SC_AVPHYS_PAGES')
    # Used memory
    used_memory = total_memory - available_memory

    # Convert to GB for readability
    total_memory_gb = round(total_memory / (1024 ** 3), 2)
    available_memory_gb = round(available_memory / (1024 ** 3), 2)
    used_memory_gb = round(used_memory / (1024 ** 3), 2)

    print(f"Total Memory: {total_memory_gb} GB")
    print(f"Available Memory: {available_memory_gb} GB")
    print(f"Used Memory: {used_memory_gb} GB")

"""
In this step a classifier is trained on the ovarian cancer patches 
to detect subtypes of ovarian cancer. Feature extractor used is H-optimus-0.
"""
# Arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, default='H-optimus-0',
                    help='name of the model')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num_epochs', type=int, default=3,
                    help="number of rounds of training")
parser.add_argument('--save_dir', type=str, default='outputs', help='path \
                        to save results')
parser.add_argument('--num_classes', type=int, default=5, help='number \
                        of classes')
parser.add_argument('--add_layer', type=str, default='True', help='add \
                        extra layer to the model for fine-tuning')
parser.add_argument('--batch_size', type=int, default=32,
                    help="local batch size: B")
parser.add_argument('--lr', type=float, default=1e-4,
                    help='learning rate')
parser.add_argument('--csv_path', type=str, default='../data')


args = parser.parse_args()

#Device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

# Data & Transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])
dataset = OvarianDataset(args, root_dir='../data', transform=transform, train='train')
dataloader = torch.utils.data.DataLoader(dataset, batch_size=args.batch_size, num_workers=4, shuffle=True)

eval_dataset = OvarianDataset(args, root_dir='../data', transform=transform, train='eval')
eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)

# Model
feature_extractor = FeatureExtractor(args, device).to(device)
classifier = nn.Linear(feature_extractor.feature_dim, args.num_classes).to(device)

# Optimizer
optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + 
                            list(classifier.parameters()), lr=args.lr)

#Loss function
criterion = nn.CrossEntropyLoss()

torch.manual_seed(args.seed) 

train_losses = []
history = {'accuracy': [], 'loss': []}

#Train model
for epoch in range(args.num_epochs):
    feature_extractor.train()
    classifier.train()

    epoch_loss = 0.0
    total_class_loss = 0.0
    total_correct = 0.0
    count = 0
    for img1, class_idx in tqdm(dataloader):
        img1, class_idx = img1.to(device), class_idx.to(device)
        if count == 0:
            print_memory_usage()
        optimizer.zero_grad()
        features = feature_extractor(img1)
        class_logits = classifier(features)
        class_loss = criterion(class_logits, class_idx)
        epoch_loss += class_loss.item()
        _, predicted = torch.max(class_logits.data, 1)
        total_correct += (predicted == class_idx).sum().item()
        count += len(class_idx)
        class_loss.backward()
        optimizer.step()
    avg_epoch_loss = epoch_loss / len(dataloader)        
    train_losses.append(avg_epoch_loss)

    history['accuracy'].append(total_correct/ count)
    history['loss'].append(avg_epoch_loss)
    print(f"Epoch {epoch}/{args.num_epochs}: Loss: {avg_epoch_loss} Acc: {total_correct/ count}")
    with open(os.path.join(args.save_dir,'train_statistics.pkl'), 'wb') as file_hist:
        pickle.dump(history, file_hist)
    print(history)
    save_checkpoint( args,
        {
            "epoch": epoch,
            "encoder": feature_extractor.state_dict(),
            "classifier": classifier.state_dict(),
            "acc": total_correct / count,
            "optimizer": optimizer.state_dict(),
        },
        save_name=f'epoch{epoch}',
        save_dir=args.save_dir,
    )
