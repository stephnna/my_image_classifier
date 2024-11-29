import os
import argparse
import torch
from torch import nn, optim
import torchvision.models as models
from utils import get_dataloaders, save_checkpoint, load_checkpoint, save_metrics_to_file


import time
from utils import set_seed
set_seed(42)

def get_model(arch, hidden_units):
    if arch == 'vgg16':
        model = models.vgg16(pretrained=True)
        
        # Freeze all layers except the last classifier
        for param in model.classifier.parameters():
            param.requires_grad = False
        # Modify the classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units // 2, 102),
            nn.LogSoftmax(dim=1)
        )

    elif arch == 'resnet101':
        model = models.resnet101(pretrained=True)
        print(model)
        return
        # Freeze all layers except the last classifier
        for param in model.classifier.parameters():
            param.requires_grad = False
        # Modify the classifier
        model.classifier = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(25088, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units // 2, 102),
            nn.LogSoftmax(dim=1)
        )

    elif arch == 'custom':
        model = nn.Sequential(
            nn.Dropout(0.6),
            nn.Linear(150528, hidden_units),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units, hidden_units // 2),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_units // 2, 102),
            nn.LogSoftmax(dim=1)
        )
    else:
        model = models.alexnet(pretrained=True)
        # Freeze all layers except the last classifier
        for param in model.classifier.parameters():
            param.requires_grad = False
        # Modify the classifier
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

    # Ensure the parameters of the new classifier layers require gradients
    for param in model.classifier.parameters():
        param.requires_grad = True

    return model


def main():
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Train a deep learning model on a dataset')
    parser.add_argument('data_dir', type=str, help='Directory containing dataset')
    parser.add_argument('arch', type=str, choices=['alexnet', 'resnet101', 'vgg16', 'custom'],  help='Model architecture')
    parser.add_argument('hidden_units', type=int, help='Number of hidden units')
    parser.add_argument('batch_size', type=int, help='Batch size')
    parser.add_argument('--save_dir', type=str, default='checkpoints', help='Directory to save the checkpoint')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for training')
    parser.add_argument('--start_epoch', type=int, default=0, help='Epoch from which to start training')
    parser.add_argument('--checkpoint', type=str, help='Path to checkpoint to resume training from (optional)')

    args = parser.parse_args()

    # Set device for GPU or CPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")

    # Initialize the model
    model = get_model(args.arch, args.hidden_units)
    return
    model.to(device)

    # Define criterion and optimizer
    criterion = nn.NLLLoss()
    optimizer = optim.Adam(model.classifier.parameters(), lr=args.learning_rate, weight_decay=0.01)

    # Get data loaders
    trainloader, _, validloader, class_to_idx = get_dataloaders(data_dir=args.data_dir, batch_size=args.batch_size)

    # Load checkpoint if provided
    if args.checkpoint and os.path.exists(args.checkpoint):
        model, optimizer, start_epoch, class_to_idx = load_checkpoint(args.checkpoint, model, optimizer)

        if args.start_epoch != start_epoch:
            print("Invalid starting epoch")
            return
        else:
            args.start_epoch = start_epoch + 1
            print(f"Resumed training  from epoch {start_epoch+1}")

    else:
        print("Checkpoint file does not exist, starting fresh training.")

    # Get a single batch of images and labels from the trainloader
    # data_iter = iter(trainloader)
    # images, labels = next(data_iter)

    # # Print the shape of images and labels
    # print("Image batch shape:", images.shape)  # Shape of images
    # print("Label batch shape:", labels.shape)  # Shape of labels

    # Train the model
    start_time = time.time()
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

            #images = images.view(images.size(0), -1)
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
                        #images = images.view(images.size(0), -1)
                        logps = model(images)
                        loss = criterion(logps, labels)

                        valid_loss += loss.item()

                        # Compute accuracy
                        ps = torch.exp(logps)
                        top_p, top_class = ps.topk(1, dim=1)
                        equals = top_class == labels.view(*top_class.shape)
                        accuracy += torch.mean(equals.float()).item()


                 # Prepare new metrics for this batch
                epoch_n_batch = [epoch+1]  # Use a list to append
                train_losses_batch = [running_loss / print_every]
                valid_losses_batch = [valid_loss / len(validloader)]
                train_acc_batch = [accuracy / len(validloader)]

                # Call the function to save metrics
                # Ensure both save_dir and experiment no subdirectory exist
                if not os.path.exists('losses/next'):
                    os.makedirs('losses/next', exist_ok=True)

                save_metrics_to_file('losses/next/losses_4.pkl', epoch_n_batch, train_losses_batch, valid_losses_batch, train_acc_batch)


                print(f"Epoch {epoch+1}/{args.epochs}.. "
                        f"Training loss: {running_loss/print_every:.3f}.. "
                        f"Validation loss: {valid_loss/len(validloader):.3f}.. "
                        f"Validation accuracy: {accuracy/len(validloader):.3f}")


                running_loss = 0
                model.train()


        # Ensure both save_dir and experiment_2 subdirectory exist
        experiment_dir = os.path.join(args.save_dir, "next/experiment_4")
        if not os.path.exists(experiment_dir):  # Check if the directory does not exist
            os.makedirs(experiment_dir, exist_ok=True)  # Create the directory if it doesn't exist


        # Save checkpoint after every epoch with file numbering
        checkpoint_path = os.path.join(experiment_dir, f"checkpoint_epoch_{epoch+1}.pth")
        save_checkpoint(model, optimizer, epoch+1, class_to_idx, checkpoint_path)

    end_time = time.time()
    computation_time = end_time - start_time
    print(f"Training completed in {computation_time:.2f} seconds.")



if __name__ == '__main__':
    main()
