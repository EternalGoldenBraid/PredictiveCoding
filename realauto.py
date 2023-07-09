import logging
log = logging.getLogger(__name__)
log.setLevel(logging.DEBUG)
log.addHandler(logging.StreamHandler())

import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as transforms
import numpy as np

from models import Autoencoder, AutoencoderEfficientnet
from utils.vis_utils import LossVisualizer

visualizer: LossVisualizer = LossVisualizer()

# Initialize the device and the model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# device = torch.device("cpu")
# model = Autoencoder().to(device)
model = AutoencoderEfficientnet().to(device)
model.eval()  # Switch the model to evaluation mode

# Load a pretrained model if available
# model.load_state_dict(torch.load("model.pth"))
# model.eval()

down_width: int = 128
down_height: int = 128

# Prepare transformations
down_sample = transforms.Resize((down_width, down_height))
to_tensor = transforms.ToTensor()
to_pil = transforms.ToPILImage()
# pre_transform = transforms.Compose([down_sample, to_tensor])
pre_transform = transforms.Compose([to_tensor])
post_transform = transforms.Compose([to_pil])

# Open the webcam
cap = cv2.VideoCapture(0)

lr: float = 0.001

# optimizer: SGD = SGD(model.parameters(), lr=lr)
optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)

while True:
    # Read a frame
    ret, frame = cap.read()
    if not ret:
        log.warning("Can't receive frame (stream end?). Exiting ...")
        break
    

    # Convert the frame to a tensor and pass it through the autoencoder
    frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    # input_tensor = to_tensor(frame)
    # input_tensor = input_tensor.view(1, -1).to(device)
    # input_tensor = pre_transform(frame).view(1, -1).to(device)
    # input_tensor = pre_transform(frame).view(1, -1)
    input_tensor = pre_transform(frame).unsqueeze(0).to(device)
    output_tensor = model(input_tensor)

    loss = F.mse_loss(input_tensor, output_tensor)
    loss.backward()
    optimizer.step()
    optimizer.zero_grad()
    
    print(f"loss: {loss}")
    visualizer.update(loss.item())

    output_frame = to_pil(output_tensor[0].cpu().detach())

    # Convert the result back to a numpy array and display it
    output_frame = np.array(output_frame)
    output_frame = cv2.cvtColor(output_frame, cv2.COLOR_RGB2BGR)

    # Compute the residual frame
    residual_frame = np.abs(np.array(frame) - output_frame)

    # Concatenate the input, output and residual frames for visualization
    frames = cv2.hconcat([np.array(frame), output_frame, residual_frame])
    
    cv2.imshow("Input | Output | Residual", frames)

    # If the user presses 'q', exit the loop
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the webcam and destroy the windows
cap.release()
cv2.destroyAllWindows()
