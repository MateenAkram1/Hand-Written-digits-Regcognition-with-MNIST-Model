import torch

def fgsm_attack(model, input, label_tensor, epsilon, loss_fn):
    input.requires_grad = True
    output = model(input)
    loss = loss_fn(output, label_tensor)    
    # Backward pass to compute gradients
    model.zero_grad()
    loss.backward()
    gradient = input.grad.data
    
    perturbation = epsilon * torch.sign(gradient)
    
    adversarial_input = input + perturbation
    adversarial_input = torch.clamp(adversarial_input, 0, 1)
    
    return adversarial_input