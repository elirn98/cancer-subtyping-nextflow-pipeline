from utils.data_loader import OvarianDataset
import torch
import torchvision
from torchvision import transforms
from torch import nn
import arg_parser
from networks import FeatureExtractor, FeatureHead
from tqdm import tqdm
import shutil
import os

def save_checkpoint(args, state, is_best, save_name, save_dir):
    save_dir = os.path.join(save_dir, args.model, 'v'+ str(args.version), 'seed'+ str(args.seed))
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = save_dir + "/" + "checkpoint_" + str(save_name) + ".pth.tar"
    torch.save(state, filename)
    if is_best:
        best_filename = save_dir + "/" + "model_best_" + str(save_name) + ".pth.tar"
        shutil.copyfile(filename, best_filename)


def evaluate_model(feature_extractor, classifier, eval_dataloader, test_dataloader, device):
    feature_extractor.eval()
    classifier.eval()
    total_test_loss = 0.0
    total_eval_loss = 0.0
    e_correct, t_correct = 0, 0
    e_total, t_total = 0, 0

    with torch.no_grad():
        for images, labels, _, _ in tqdm(eval_dataloader): 
            images = images.to(device)
            labels = labels.to(device)
            
            features = feature_extractor(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            total_eval_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            e_total += labels.size(0)
            e_correct += (predicted == labels).sum().item()
    avg_eval_loss = total_eval_loss / len(eval_dataloader)
    e_accuracy = e_correct / e_total

    
    with torch.no_grad():
        for images, labels, _, _ in tqdm(test_dataloader): 
            images = images.to(device)
            labels = labels.to(device)
            
            features = feature_extractor(images)
            outputs = classifier(features)
            loss = criterion(outputs, labels)
            
            total_test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            t_total += labels.size(0)
            t_correct += (predicted == labels).sum().item()
    
    avg_test_loss = total_test_loss / len(test_dataloader)
    t_accuracy = t_correct / t_total
    return avg_eval_loss, e_accuracy, avg_test_loss, t_accuracy


if __name__ == "__main__":
    # Arguments
    args = arg_parser.args_parser()

    #Device 
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(device)
    print(args.model)

    # Data
    transform = transforms.Compose([
        # transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.CenterCrop(224),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    dataset = OvarianDataset(args, root_dir='./data', transform=transform, train='train', saved_features=args.saved_features)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=64, num_workers=4, shuffle=True)

    eval_dataset = OvarianDataset(args, root_dir='./data', transform=transform, train='eval', saved_features=args.saved_features)
    eval_dataloader = torch.utils.data.DataLoader(eval_dataset, batch_size=32, num_workers=4, shuffle=False)

    test_dataset = OvarianDataset(args, root_dir='./data', transform=transform, train='test', saved_features=args.saved_features)
    test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size=32, num_workers=4, shuffle=False)
    # Initialize components
    if args.saved_features == 'True':
        feature_extractor = FeatureHead(args).to(device)
    else:
        feature_extractor = FeatureExtractor(args, device).to(device)

    classifier = nn.Linear(feature_extractor.feature_dim, args.num_classes).to(device)

    # Optimizer
    optimizer = torch.optim.Adam(list(feature_extractor.parameters()) + 
                                list(classifier.parameters()), lr=args.lr)
    criterion = nn.CrossEntropyLoss()

    train_losses = []
    eval_losses = []
    test_losses = []
    best_acc = 0.0
    best_step = 0

    torch.manual_seed(args.seed) 
    print('seed', args.seed)

    for epoch in range(args.num_epochs):
        feature_extractor.train()
        classifier.train()

        epoch_loss = 0.0
        total_class_loss = 0.0
        for img1, class_idx, slide_id in tqdm(dataloader):
            img1, class_idx = img1.to(device), class_idx.to(device)
            optimizer.zero_grad()
            # Extract features and decompose into domain-generic and domain-specific features
            features = feature_extractor(img1)
           
            # Use combined features for classification
            class_logits = classifier(features)
            class_loss = criterion(class_logits, all_class_idx)
            
            epoch_loss += class_loss.item()
            class_loss.backward()
            optimizer.step()
                   
        avg_epoch_loss = epoch_loss / len(dataloader)
        print(f"Epoch {epoch}/{args.num_epochs}: Loss: {avg_epoch_loss}")
        
        eval_loss, eval_accuracy, test_loss, test_accuracy = evaluate_model(feature_extractor, classifier, eval_dataloader, test_dataloader, device)
        # test domain specific features domain classification accuracy
        train_losses.append(avg_epoch_loss)
        eval_losses.append(eval_loss)
        test_losses.append(test_loss)
        print(f"Epoch {epoch}/{args.num_epochs}: Eval Loss: {eval_loss}, Eval Accuracy: {eval_accuracy}")
        print(f"Epoch {epoch}/{args.num_epochs}: Test Loss: {test_loss}, Test Accuracy: {test_accuracy}")
        is_best = eval_accuracy > best_acc
        best_acc = max(eval_accuracy, best_acc)
        if is_best:
            best_step = epoch
        save_checkpoint( args,
            {
                "epoch": epoch,
                "encoder": feature_extractor.state_dict(),
                "classifier": classifier.state_dict(),
                "best_acc": best_acc,
                "optimizer": optimizer.state_dict(),
            },
            is_best,
            save_name=f'epoch{epoch}',
            save_dir=args.save_dir,
        )
