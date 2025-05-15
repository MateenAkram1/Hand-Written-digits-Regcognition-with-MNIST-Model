import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
from mnist_case_cnn import MNISTCaseCNN
import matplotlib.pyplot as plt

def train_and_test_with_plots(case_number):
    # Settings
    batch_size = 100
    epochs = 15 
    learning_rate = 0.01

    # Choose GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Load MNIST dataset
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train_data = datasets.MNIST(root='./data', train=True, transform=transform, download=True)
    test_data = datasets.MNIST(root='./data', train=False, transform=transform, download=True)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=False)

    # Initialize model, loss, optimizer
    model = MNISTCaseCNN(case=case_number).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # Lists to store metrics
    train_losses, train_accuracies = [], []
    val_losses, val_accuracies = [], []

    # Training loop
    for epoch in range(1, epochs + 1):
        model.train()
        running_loss, correct, total = 0.0, 0, 0
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * images.size(0)
            preds = outputs.argmax(dim=1)
            correct += (preds == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = 100 * correct / total
        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_acc)

        # Validation
        model.eval()
        val_running_loss, val_correct, val_total = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                loss = criterion(outputs, labels)
                val_running_loss += loss.item() * images.size(0)
                preds = outputs.argmax(dim=1)
                val_correct += (preds == labels).sum().item()
                val_total += labels.size(0)

        val_epoch_loss = val_running_loss / val_total
        val_epoch_acc = 100 * val_correct / val_total
        val_losses.append(val_epoch_loss)
        val_accuracies.append(val_epoch_acc)

        print(f"Epoch {epoch:2d}: Train Loss={epoch_loss:.4f}, Train Acc={epoch_acc:.2f}%, "
              f"Val Loss={val_epoch_loss:.4f}, Val Acc={val_epoch_acc:.2f}%")

    # Plot training & validation loss
    plt.figure()
    plt.plot(range(1, epochs+1), train_losses, label='Train Loss')
    plt.plot(range(1, epochs+1), val_losses, label='Validation Loss')
    plt.title(f'Case {case_number} Loss over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.show()

    # Plot training & validation accuracy
    plt.figure()
    plt.plot(range(1, epochs+1), train_accuracies, label='Train Accuracy')
    plt.plot(range(1, epochs+1), val_accuracies, label='Validation Accuracy')
    plt.title(f'Case {case_number} Accuracy over Epochs')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy (%)')
    plt.legend()
    plt.show()

    # Display one test image and its predicted output
    model.eval()
    with torch.no_grad():
        sample_img, sample_label = next(iter(test_loader))
        sample_img, sample_label = sample_img[0].unsqueeze(0).to(device), sample_label[0].item()
        output = model(sample_img)
        pred_label = output.argmax(dim=1).item()

    # Convert image for plotting
    img_to_show = sample_img.cpu().squeeze().numpy()

    plt.figure()
    plt.imshow(img_to_show, cmap='gray', interpolation='nearest')
    plt.title(f'Ground Truth: {sample_label} | Predicted: {pred_label}')
    plt.axis('off')
    plt.show()

if __name__ == "__main__":
    case = int(input("Enter Case Number (1-6): "))
    train_and_test_with_plots(case)
