# 🌸 Flower Species Image Classifier

This project is a Deep Learning Image Classifier built using PyTorch to identify different flower species. It supports architectures like VGG16, ResNet101, and AlexNet, allowing for accurate flower species predictions.

## 🚀 Features
- ✨ **Multiple Architectures**: VGG16, ResNet101, AlexNet, and custom models.
- ✨ **Transfer Learning**: Leverages pre-trained models for better performance.
- ✨ **Checkpointing**: Save and resume training seamlessly.
- ✨ **Device Compatibility**: Works on both CPU and GPU.
- ✨ **Top-K Predictions**: Get probabilities for the top-K most likely classes.
- ✨ **Customizable Training**: Configurable hyperparameters for flexibility.

## 🛠️ Installation
### Clone the repository:
```bash
git clone https://github.com/your-username/my_image_classifier.git
cd my_image_classifier

Install dependencies:
pip install -r requirements.txt

Install PyTorch (if not already installed):
Follow PyTorch installation instructions here.

🏋️‍♀️ Training the Model
Prepare your dataset:
Organize your data directory like this:

data/
├── train/
├── valid/
└── test/

Run the training script:

python train.py <data_dir> <arch> <hidden_units> <batch_size> --gpu --epochs <epochs> --learning_rate <lr>


Example: python train.py flowers vgg16 512 64 --gpu --epochs 10 --learning_rate 0.001

Save checkpoints:
Checkpoints will be saved in the checkpoints/ directory automatically.

🔍 Prediction
Run the prediction script:

python predict.py <image_path> <checkpoint_path> --top_k <K> --category_names <json_path> --gpu


Output:
Top Predictions:

Sunflower (87%)
Daisy (5%)
Rose (3%)
📂 Project Structure


File/Folder	Description
train.py	Script for training the classifier.
predict.py	Script for making predictions on new images.
utils.py	Helper functions for data loading, checkpointing, and preprocessing.
data/	Dataset directory.
checkpoints/	Directory for saved model checkpoints.
losses/	Logs for tracking training/validation losses and accuracy.

📊 Metrics Tracking
Training Loss: Measures the model's error during training.
Validation Accuracy: Evaluates the model's performance on unseen data.
Saved Logs: Automatically stored in the losses/ directory.
📝 Example Workflow

Train the model:
python train.py flowers vgg16 512 64 --gpu --epochs 10

Save the checkpoint:
The checkpoint is saved as checkpoints/vgg16_checkpoint.pth.

Predict a flower class:

python predict.py flowers/test/1/image_06752.jpg checkpoints/vgg16_checkpoint.pth --top_k 3 --gpu


💡 Future Enhancements
Add support for additional architectures.
Implement a GUI or web interface for easier predictions.
Optimize performance for larger datasets.
🤝 Contributing
Contributions are welcome! Follow these steps to contribute:

Fork the repository.

git checkout -b feature-name

Commit your changes
git commit -m "Add new feature"

Push to your branch:
git push origin feature-name

🧑‍💻 Author
Stephen Ezea
GitHub: github.com/stephnna/
LinkedIn: https://www.linkedin.com/in/stephen-ezea

🌟 Acknowledgments
Special thanks to:

The PyTorch community for their amazing library.
Amazon for datasets and inspiration.
🏵️ Happy Coding & Flower Classifying! 🏵️

