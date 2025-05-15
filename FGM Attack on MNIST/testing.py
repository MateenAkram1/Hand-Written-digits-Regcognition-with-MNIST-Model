import torch
from torchvision import transforms, datasets  
from torchvision.datasets import MNIST  
import matplotlib.pyplot as plt
import fgsm
import fgsm_gaussian
from mnist_model import MNIST_CNN
import numpy as np
# Load model
model = MNIST_CNN()
model.load_state_dict(torch.load('mnist_model.pth'))
model.eval()

# Load dataset
transform = transforms.Compose([transforms.ToTensor()])
test_dataset = MNIST(root='./data', train=False, download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=1, shuffle=True)

def compare_attack_results(model, images, labels, attack_fn, epsilon):
    """Compare original and adversarial examples visually and numerically"""
    adv_images = attack_fn(model, images, labels, epsilon, torch.nn.CrossEntropyLoss())
    
    # Get predictions
    with torch.no_grad():
        orig_output = model(images)
        adv_output = model(adv_images)
    
    orig_pred = torch.argmax(orig_output).item()
    adv_pred = torch.argmax(adv_output).item()
    
    # Visualization
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 3, 1)
    plt.imshow(images.squeeze().detach().numpy(), cmap='gray')
    plt.title(f'Original\nPred: {orig_pred}')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(adv_images.squeeze().detach().numpy(), cmap='gray')
    plt.title(f'Adversarial (ε={epsilon})\nPred: {adv_pred}')
    plt.axis('off')
    
    # Perturbation (only for ε > 0)
    plt.subplot(1, 3, 3)
    if epsilon > 0:
        difference = (adv_images - images).squeeze().detach().numpy()
        plt.imshow(difference, cmap='seismic', 
                  vmin=-epsilon, vmax=epsilon,
                  interpolation='nearest')
        plt.colorbar(ticks=[-epsilon, 0, epsilon])
        plt.title(f'Perturbation (Max: {np.abs(difference).max():.4f})')
    else:
        plt.text(0.5, 0.5, 'No perturbation (ε=0)', 
                 ha='center', va='center')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
def evaluate_attack(model, test_loader, attack_fn, epsilon, num_samples=5):
    """Evaluate attack and show examples"""
    correct = 0
    total = 0
    samples_shown = 0
    
    for images, labels in test_loader:
        adv_images = attack_fn(model, images, labels, epsilon, torch.nn.CrossEntropyLoss())
        outputs = model(adv_images)
        _, predicted = torch.max(outputs.data, 1)
        
        total += labels.size(0)
        correct += (predicted == labels).sum().item()
        
        if samples_shown < num_samples:
            compare_attack_results(model, images, labels, attack_fn, epsilon)
            samples_shown += 1
    
    accuracy = 100 * correct / total
    return accuracy

# Evaluate attacks
epsilon = 0.1
print("Evaluating clean accuracy...")
clean_accuracy = evaluate_attack(model, test_loader, lambda *args: args[1], 0)

print("\nEvaluating FGSM attack...")
fgsm_accuracy = evaluate_attack(model, test_loader, fgsm.fgsm_attack, epsilon)

print("\nEvaluating Gaussian FGSM attack...")
gaussian_accuracy = evaluate_attack(model, test_loader, fgsm_gaussian.fgsm_gaussian_attack, epsilon)

# Final results
print("\nFinal Results:")
print(f"Clean Accuracy: {clean_accuracy:.2f}%")
print(f"FGSM Accuracy: {fgsm_accuracy:.2f}%")
print(f"Gaussian FGSM Accuracy: {gaussian_accuracy:.2f}%")