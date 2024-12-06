from utils.data_loader import OvarianDataset
import torch
import torchvision
from torchvision import transforms
from torch import nn
from networks import FeatureExtractor, FeatureHead
from tqdm import tqdm
import shutil
import os
import argparse
import pickle


def test_model(feature_extractor, criterion, classifier, test_dataloader, device):
    # This Function evaluate the performance of model trained in last step.
    feature_extractor.eval()
    classifier.eval()
    total_test_loss = 0.0
    total_eval_loss = 0.0
    t_correct = 0
    t_total = 0
    test_results = {'label_test': [], 'label_prediction': []}
 
    with torch.no_grad():
        for images, labels in tqdm(test_dataloader): 
            images = images.to(device)
            labels = labels.to(device)
            features = feature_extractor(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            t_total += labels.size(0)
            t_correct += (predicted == labels).sum().item()
            test_results['label_prediction'].extend(predicted)
            test_results['label_test'].extend(labels)
    
    avg_test_loss = total_test_loss / len(test_dataloader)
    t_accuracy = t_correct / t_total
    return avg_test_loss, t_accuracy, test_results

#Arguments
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--model', type=str, default='H-optimus-0',
                    help='name of the model')
parser.add_argument('--seed', type=int, default=42, help='random seed')
parser.add_argument('--num_classes', type=int, default=5, help='number \
                        of classes')
parser.add_argument('--model_path', type=str, default='./outputs/model.pth.tar', help='path to the model')
parser.add_argument('--csv_path', type=str, default='../data')
parser.add_argument('--add_layer', type=str, default='True', help='add \
                        extra layer to the model for fine-tuning')
parser.add_argument('--save_dir', type=str, default='outputs', help='path \
                        to save results')
parser.add_argument('--batch_size', type=int, default=16,
                    help="local batch size: B")

args = parser.parse_args()

#Device 
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Data
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.CenterCrop(224),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_dataset = OvarianDataset(args, root_dir='../data', transform=transform, train='test')
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, num_workers=4, shuffle=False)


# Loss function
criterion = nn.CrossEntropyLoss()

# Load fine-tuned model
checkpoint = torch.load(args.model_path)

#Initialize components
feature_extractor = FeatureExtractor(args, device).to(device)
feature_extractor.load_state_dict(checkpoint['encoder'])

classifier = nn.Linear(feature_extractor.feature_dim, args.num_classes).to(device)
classifier.load_state_dict(checkpoint['classifier'])

#Evaluate model
t_loss, t_acc, test_results = test_model(feature_extractor, criterion, classifier, test_dataloader, device)

with open(os.path.join(args.save_dir,'test_results.pkl'), 'wb') as f:
    pickle.dump(test_results, f)

print(f"Test Loss: {t_loss}, Test Accuracy: {t_acc}")
