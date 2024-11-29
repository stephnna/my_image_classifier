import torch
from torch import nn, optim
from utils import get_dataloaders, load_checkpoint
from train import get_model

from utils import set_seed
set_seed(42)

model = get_model()
optimizer = optim.Adam(model.parameters(), lr=0.001)
# Load the trained model from the checkpoint
model, _, _, _ = load_checkpoint("checkpoints/checkpoint_epoch_10.pth", model, optimizer)

# Set device for GPU or CPU
device = torch.device("cuda" if "gpu" and torch.cuda.is_available() else "cpu")
model.to(device)

criterion = nn.NLLLoss()
# Switch the model to evaluation mode
model.eval()

# Initialize variables to track test loss and accuracy
test_loss = 0
accuracy = 0

_, testloader, _, class_to_idx = get_dataloaders(data_dir="flowers", batch_size=8)
# Turn off gradients for testing

with torch.no_grad():
    for images, labels in testloader:
        # Move images and labels to GPU)
        images, labels = images.to(device), labels.to(device)
        
        # get predictions from the model
        log_ps = model(images)
        
        # Compute the test loss
        loss = criterion(log_ps, labels)
        
        test_loss += loss.item()  
        ps = torch.exp(log_ps)  
        top_p, top_class = ps.topk(1, dim=1)  
      
        equals = top_class == labels.view(*top_class.shape) 
        accuracy += torch.mean(equals.type(torch.FloatTensor)).item()


test_loss = test_loss / len(testloader)
accuracy = accuracy / len(testloader)


print(f"Test Loss: {test_loss:.3f}.. "
      f"Test Accuracy: {accuracy * 100:.3f}%")
