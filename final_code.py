import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import random
import matplotlib.pyplot as plt

# Step 1: Generate Synthetic Data
def generate_synthetic_data(num_samples=1000, room_size_range=(10, 20)):
    data = []
    for _ in range(num_samples):
        room_w = random.randint(*room_size_range)
        room_h = random.randint(*room_size_range)
        
        furniture = [
            {"type": "bed", "w": random.randint(4, 6), "h": random.randint(6, 8)},
            {"type": "desk", "w": random.randint(2, 4), "h": random.randint(3, 5)},
            {"type": "sofa", "w": random.randint(5, 7), "h": random.randint(3, 5)},
            {"type": "table", "w": random.randint(3, 5), "h": random.randint(3, 4)}
        ]
        
        placements = []
        occupied = []  # Track occupied areas
        for f in furniture:
            for _ in range(50):  # Try multiple placements
                x = random.randint(0, room_w - f["w"])
                y = random.randint(0, room_h - f["h"])
                if all(not (ox <= x < ox + ow and oy <= y < oy + oh) for (ox, oy, ow, oh) in occupied):
                    occupied.append((x, y, f["w"], f["h"]))
                    placements.append((x / room_w, y / room_h))  # Normalize
                    break
        
        if len(placements) == len(furniture):  # Ensure valid placement
            data.append(((room_w, room_h) + tuple(f["w"] for f in furniture) + tuple(f["h"] for f in furniture), placements))
    
    return data

# Step 2: Define Neural Network
class FurniturePlacementNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(FurniturePlacementNN, self).__init__()
        self.fc1 = nn.Linear(input_size, 64)
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, output_size)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.fc3(x)
        return x

# Step 3: Training

data = generate_synthetic_data()
inputs = torch.tensor([d[0] for d in data], dtype=torch.float32)
outputs = torch.tensor([np.array(d[1]).flatten() for d in data], dtype=torch.float32)

model = FurniturePlacementNN(inputs.shape[1], outputs.shape[1])
criterion = nn.MSELoss()
optimizer = optim.Adam(model.parameters(), lr=0.01)

for epoch in range(1000):
    optimizer.zero_grad()
    predictions = model(inputs)
    loss = criterion(predictions, outputs)
    loss.backward()
    optimizer.step()
    if epoch % 100 == 0:
        print(f'Epoch {epoch}, Loss: {loss.item()}')

# Step 4: Visualization

def visualize_layout(room_w, room_h, furniture, positions):
    fig, ax = plt.subplots()
    ax.set_xlim(0, room_w)
    ax.set_ylim(0, room_h)
    for (f, pos) in zip(furniture, positions):
        rect = plt.Rectangle((pos[0] * room_w, pos[1] * room_h), f["w"], f["h"], edgecolor='black', facecolor='lightblue')
        ax.add_patch(rect)
        ax.text(pos[0] * room_w + f["w"] / 2, pos[1] * room_h + f["h"] / 2, f["type"], ha='center', va='center')
    plt.show()

# Testing visualization with one sample
test_sample = random.choice(data)
visualize_layout(test_sample[0][0], test_sample[0][1], [
    {"type": "bed", "w": test_sample[0][2], "h": test_sample[0][6]},
    {"type": "desk", "w": test_sample[0][3], "h": test_sample[0][7]},
    {"type": "sofa", "w": test_sample[0][4], "h": test_sample[0][8]},
    {"type": "table", "w": test_sample[0][5], "h": test_sample[0][9]}],
    test_sample[1])
