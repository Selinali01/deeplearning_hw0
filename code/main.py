import torch
from torch.utils.data import DataLoader
from lenet5 import LeNet5
from load_data import load_mnist_data
from train import train
from evaluate import evaluate
from matplotlib import pyplot as plt

def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Load data
    train_dataset, val_dataset, test_dataset = load_mnist_data()
    
    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)
    
    # Initialize model
    use_dropout = False  # Toggle this to True/False to enable/disable dropout
    use_batchnorm = False  # Toggle this to True/False to enable/disable batch normalization
    weight_decay = 0.000  # Set this to 0 to disable weight decay
    
    model = LeNet5(use_dropout=use_dropout, use_batchnorm=use_batchnorm).to(device)
    
    # Train model
    num_epochs = 10
    learning_rate = 0.001
    trained_model, train_acc, val_acc = train(model, train_loader, val_loader, num_epochs, learning_rate, device, weight_decay)
    
    # Evaluate on test set
    test_accuracy = evaluate(trained_model, test_loader, device)
    print(f"Test Accuracy: {test_accuracy:.2f}%")

    # Convergence Graph
    plt.figure(figsize=(10, 5))
    plt.plot(train_acc, label="Train Accuracy")
    plt.plot(val_acc, label="Validation Accuracy")
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.title(f'Train and Validation Accuracy\n(Dropout: {use_dropout}, BatchNorm: {use_batchnorm}, Weight Decay: {weight_decay})')
    plt.legend()
    plt.savefig(f'convergence_dropout{use_dropout}_batchnorm{use_batchnorm}_weightdecay{weight_decay}.png')
    plt.show()

if __name__ == "__main__":
    main()