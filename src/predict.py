"""Classify a single image as stop / not_stop using the trained model."""
import argparse

import torch
import torchvision.models as models
from PIL import Image
from torchvision import transforms

CLASS_NAMES = ["not_stop", "stop"]  # alphabetical order, same as ImageFolder


def load_model(weights_path):
    model = models.resnet18(weights=None)
    model.fc = torch.nn.Linear(model.fc.in_features, len(CLASS_NAMES))
    model.load_state_dict(torch.load(weights_path, map_location="cpu"))
    model.eval()
    return model


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("image")
    parser.add_argument("--weights", default="model.pt")
    args = parser.parse_args()

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
    ])
    image = Image.open(args.image).convert("RGB")
    x = transform(image).unsqueeze(0)  # add batch dimension

    model = load_model(args.weights)
    with torch.no_grad():
        probs = torch.softmax(model(x), dim=1)[0]
    idx = int(probs.argmax())
    print(f"{args.image}: {CLASS_NAMES[idx]} ({probs[idx].item():.1%})")


if __name__ == "__main__":
    main()
