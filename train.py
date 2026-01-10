import torch
import torch.optim as optim
from torchvision import datasets, transforms
from model import ConvNet
import os

# Configuration
BATCH_SIZE = 64
EPOCHS = 5  # Increased for better accuracy
LEARNING_RATE = 1.0
SAVE_PATH = "mnist_model.pt"

def run_training():
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    
    print(f"Training started on: {device}")

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    try:
        train_data = datasets.MNIST('./data', train=True, download=True, transform=transform)
        train_loader = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
    except Exception as e:
        return f"Error loading data: {e}"

    model = ConvNet().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=LEARNING_RATE)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.7)

    model.train()
    for epoch in range(1, EPOCHS + 1):
        print(f"Epoch {epoch}/{EPOCHS}...")
        for batch_idx, (data, target) in enumerate(train_loader):
            data, target = data.to(device), target.to(device)
            optimizer.zero_grad()
            output = model(data)
            loss = torch.nn.functional.nll_loss(output, target)
            loss.backward()
            optimizer.step()
        scheduler.step()

    torch.save(model.state_dict(), SAVE_PATH)
    print("Training finished.")
    return "Training Complete! Model saved."

if __name__ == "__main__":
    run_training()