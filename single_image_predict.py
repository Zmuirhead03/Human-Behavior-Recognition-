import os
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models, transforms
from PIL import Image, ImageTk
import tkinter as tk
from tkinter import filedialog, messagebox
import matplotlib.pyplot as plt


modelPath = r"C:\AI_Project\stanford40_resnet18.pth"

classNames = [
    'applauding', 'blowing_bubbles', 'brushing_teeth', 'cleaning_the_floor',
    'climbing', 'cooking', 'cutting_trees', 'cutting_vegetables',
    'drinking', 'feeding_a_horse', 'fishing', 'fixing_a_bike',
    'fixing_a_car', 'gardening', 'holding_an_umbrella',
    'jumping', 'looking_through_a_microscope',
    'looking_through_a_telescope', 'phoning', 'playing_guitar',
    'playing_violin', 'pouring_liquid', 'pushing_a_cart', 'reading',
    'riding_a_bike', 'riding_a_horse', 'rowing_a_boat', 'running',
    'shooting_an_arrow', 'smoking', 'taking_photos', 'texting_message',
    'throwing_frisby', 'using_a_computer', 'walking_the_dog',
    'washing_dishes', 'watching_TV', 'waving_hands',
    'writing_on_a_board', 'writing_on_a_book'
]

validExts = [".jpg", ".jpeg", ".png", ".bmp"]
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def loadModel():
    numClasses = len(classNames)

    model = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
    inFeatures = model.fc.in_features
    model.fc = nn.Linear(inFeatures, numClasses)

    stateDict = torch.load(modelPath, map_location=device)
    model.load_state_dict(stateDict)

    model = model.to(device)
    model.eval()

    return model


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std =[0.229, 0.224, 0.225]
    )
])


def isImageFile(filePath):
    lower = filePath.lower()
    for ext in validExts:
        if lower.endswith(ext):
            return True
    return False


def predictImage(model, filePath):
    img = Image.open(filePath).convert("RGB")

    x = transform(img)
    x = x.unsqueeze(0).to(device)

    with torch.no_grad():
        outputs = model(x)

        _, predIdxTensor = torch.max(outputs, 1)
        predIdx = int(predIdxTensor.item())
        predLabel = classNames[predIdx]

        probTensor = F.softmax(outputs, dim=1)
        predProb = float(probTensor[0, predIdx].item() * 100.0)

    return img, predLabel, predProb


def main():
    model = loadModel()

    root = tk.Tk()
    root.title("Stanford-40 Image Predictor")
    root.geometry("800x500")
    root.configure(bg="#222222")

    imgLabel = tk.Label(root, bg="#222222")
    imgLabel.pack(side="left", padx=20, pady=20)

    predictionLabel = tk.Label(root, text="Select an image to begin.", 
                               font=("Arial", 16), fg="white", bg="#222222",
                               justify="left")
    predictionLabel.pack(side="top", padx=20, pady=20)

    def onSelectImage():
        filePath = filedialog.askopenfilename(title="Select Image")

        if not filePath:
            return
        if not isImageFile(filePath):
            messagebox.showerror("Error", "Please select a valid image.")
            return

        img, predLabel, predProb = predictImage(model, filePath)

        displayImg = img.resize((350, 350))
        tkImg = ImageTk.PhotoImage(displayImg)
        imgLabel.config(image=tkImg)
        imgLabel.image = tkImg

        predictionText = f"Prediction: {predLabel}\nConfidence: {predProb:.2f}%"
        predictionLabel.config(text=predictionText)

    selectButton = tk.Button(root, text="Select Image", command=onSelectImage,
                             font=("Arial", 14), bg="#444444", fg="white")
    selectButton.pack(side="bottom", pady=20)

    root.mainloop()


if __name__ == "__main__":
    main()