import argparse
import torch
from torch import optim
from utils import process_image, load_checkpoint, load_category_names
from train import get_model

from utils import set_seed
set_seed(42)


def predict(image_path, model, topk=5, device='cpu'):
    ''' Predict the class (or classes) of an image using a trained model. '''

    # Process the image
    img = process_image(image_path)

    # Convert to PyTorch tensor
    img_tensor = torch.from_numpy(img).type(torch.FloatTensor).unsqueeze(0)

    # Move the model and input tensor to the same device (GPU if available)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    img_tensor = img_tensor.to(device) 

    # Make prediction
    model.eval()
    with torch.no_grad():
        logps = model(img_tensor)

    ps = torch.exp(logps)
    top_probs, top_indices = ps.topk(topk, dim=1)

    # Convert indices to class labels
    top_probs = top_probs.cpu().numpy().squeeze()
    top_indices = top_indices.cpu().numpy().squeeze()


    idx_to_class = model.class_to_idx
    top_classes = [key for idx in top_indices for key, value in idx_to_class.items() if value == idx]

    return top_probs, top_classes

def main():
    parser = argparse.ArgumentParser(description='Predict flower name from an image')
    parser.add_argument('image_path', type=str, help='Path to the image file')
    parser.add_argument('checkpoint', type=str, help='Path to the checkpoint file')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top predictions to return')
    parser.add_argument('--category_names', type=str, default=None, help='Path to a JSON file mapping categories to names')
    parser.add_argument('--gpu', action='store_true', help='Use GPU for inference')

    args = parser.parse_args()

    model = get_model()
    optimizer = optim.Adam(model.classifier.parameters(), lr=0.001)

    # Load the trained model from the checkpoint
    model, _, _, _ = load_checkpoint(args.checkpoint, model, optimizer)

    # Set device for GPU or CPU
    device = torch.device("cuda" if args.gpu and torch.cuda.is_available() else "cpu")
    model.to(device)

    # Make prediction
    probs, classes = predict(args.image_path, model, topk=args.top_k, device=device)

    # Convert the class indices to actual names
    if args.category_names:
        cat_to_name = load_category_names('cat_to_name.json')
        classes = [cat_to_name[str(cls)] for cls in classes]
        
    # Print the top k predictions
    for i in range(len(probs)):
        print(f"Prediction: {classes[i]}, Probability: {probs[i]:.4f}")

if __name__ == '__main__':
    main()
