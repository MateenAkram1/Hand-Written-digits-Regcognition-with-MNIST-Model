from fastapi import FastAPI, File, UploadFile, Form, Query
from fastapi.responses import JSONResponse
import torch
from PIL import Image
import io
from fgsm import fgsm_attack
from torchvision import transforms
from mnist_model import MNIST_CNN  
from PIL import Image
import io

app = FastAPI()

try:
    model = MNIST_CNN()
    model.load_state_dict(torch.load('mnist_model.pth', map_location=torch.device('cpu')))
    model.eval()
except Exception as e:
    print(f"Model loading error: {str(e)}")
    raise

def preprocess_image(image_bytes):
    image = Image.open(io.BytesIO(image_bytes)).convert('L').resize((28, 28))
    transform = transforms.Compose([
        transforms.ToTensor(),
    ])
    return transform(image).unsqueeze(0)


@app.post("/attack")
async def attack(
    image: UploadFile = File(...),
    label: int = Query(...),  
    epsilon: float = Form(0.1)
):
    label_tensor = torch.tensor([label], dtype=torch.long)
    
    image_bytes = await image.read()
    input_tensor = preprocess_image(image_bytes)
    
    adversarial_input = fgsm_attack(model, input_tensor, label_tensor, epsilon, torch.nn.CrossEntropyLoss())
    
    # Check prediction
    with torch.no_grad():
        output = model(adversarial_input)
        predicted = torch.argmax(output).item()
    
    return {"success": predicted != label}