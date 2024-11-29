import os
import argparse
import torch
from torch import nn, optim
import torchvision.models as models
from utils import get_dataloaders, save_checkpoint, load_checkpoint

import pickle


def get_model(arch='alexnet', hidden_units=256):
    if arch == 'vgg13':
        model = models.vgg13(pretrained=True)
    else:
        model = models.alexnet(pretrained=True)

    # Freeze parameters to avoid backprop
    for param in model.parameters():
        param.requires_grad = False

    # Model architecture
    if arch == 'vgg13':
        model.classifier = nn.Sequential(
            nn.Linear(25088, hidden_units), 
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units, hidden_units // 2),  
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units // 2, 102),
            nn.LogSoftmax(dim=1)
        )

    else:
        model.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(9216, hidden_units),  
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units // 2, 102),
            nn.LogSoftmax(dim=1)
        )

    return model

# Function to save metrics to a file
def save_metrics_to_file(filename, epoch_n, train_losses, valid_losses, train_acc):
    # Check if the file already exists
    if os.path.exists(filename):
        # Load existing data
        with open(filename, 'rb') as f:
            data = pickle.load(f)
            # Append new metrics to existing lists
            data['epoch_n'].extend(epoch_n)
            data['train_losses'].extend(train_losses)
            data['valid_losses'].extend(valid_losses)
            data['train_acc'].extend(train_acc)
    else:
        # If the file doesn't exist, create a new dictionary
        data = {
            'epoch_n': epoch_n,
            'train_losses': train_losses,
            'valid_losses': valid_losses,
            'train_acc': train_acc
        }

    # Save the combined data to the file
    with open(filename, 'wb') as f:
        pickle.dump(data, f)


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a deep learning model on a dataset')
    parser.add_argument('data_dir', type=str, help='Directory containing dataset')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save the checkpoint')
    parser.add_argument('--arch', type=str, choices=['alexnet', 'vgg13'], default='alexnet', help='Model architecture')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--hidden_units', type=int, default=256, help='Number of hidden units')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch from which to start training')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training from (optional)')

    args = parser.parse_args()

    # Set device for GPU or CPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = get_model(args.arch, args.hidden_units)
    model.to(device)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Get data loaders
    trainloader, _, validloader, class_to_idx = get_dataloaders(data_dir=args.data_dir, batch_size=16)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, optimizer, start_epoch, class_to_idx = load_checkpoint(args.checkpoint, model, optimizer)

        if args.start_epoch != start_epoch+1:
            print("Invalid starting epoch")
            return
        else:
            print(f"Resumed training  from epoch {start_epoch+1}")

    else:
        print("Checkpoint file does not exist, starting fresh training.")  

    # Train the model

    steps = 0
    running_loss = 0
    print_every = 5
    for epoch in range(args.start_epoch, args.epochs):
        for images, labels in trainloader:
            steps += 1
            # Move input and label tensors to the default device
            images, labels = images.to(device), labels.to(device)

            # Turn off gradients
            optimizer.zero_grad()

            # Make a forward pass
            logps = model(images)
            # Calculate loss
            loss = criterion(logps, labels)
            # Perform back propagation
            loss.backward()
            # weight step
            optimizer.step()

            running_loss += loss.item()

            if steps % print_every == 0:
                valid_loss = 0
                accuracy = 0
                model.eval()
                with torch.no_grad():
                    for images, labels in validloader:
                        images, labels = images.to(device), labels.to(device)
                        logps = model(images)
                        loss = criterion(logps, labels)
                        
                        valid_loss += loss.item()
                        
                        # Compute accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.float()).item()


                 # Prepare new metrics for this batch
                epoch_n_batch = [epoch + 1]  # Use a list to append
                train_losses_batch = [running_loss / len(trainloader)]
                valid_losses_batch = [valid_loss / len(validloader)]
                train_acc_batch = [accuracy / len(validloader)]

                # Call the function to save metrics
                save_metrics_to_file('losses.pkl', epoch_n_batch, train_losses_batch, valid_losses_batch, train_acc_batch)


                print(f"Epoch {epoch}/{args.epochs}.. "
                        f"Training loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validloader):.3f}")

                running_loss = 0
                model.train()    


        # Ensure save_dir is a directory
        if not os.path.exists(args.save_dir):
            os.makedirs(args.save_dir)

        # Save checkpoint after every epoch with file numbering
        checkpoint_path = os.path.join(args.save_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch+1,  class_to_idx, checkpoint_path)

if __name__ == '__main__':
    main()
