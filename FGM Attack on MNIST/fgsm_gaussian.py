import torch

def fgsm_gaussian_attack(model, input, label, epsilon, loss_fn):
    input.requires_grad = True
    
    # Forward pass
    output = model(input)
    loss = loss_fn(output, label)
    
    # Backward pass to compute gradients
    model.zero_grad()
    loss.backward()
    gradient = input.grad.data
    
    noise = torch.randn_like(input) * epsilon 
    adversarial_input = input + noise
    adversarial_input = torch.clamp(adversarial_input, 0, 1)
    
    return adversarial_input