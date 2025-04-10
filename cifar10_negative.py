# necessary imports
import torch
from openood.evaluation_api import Evaluator
from openood.networks import ResNet18_32x32

# Initialize the network
network = ResNet18_32x32()

# Load the checkpoint
checkpoint_path = './Out_of_Distribution/Graph_stuff/cifar10_resnet18_32x32_base_e100_lr0.1_default/s0/best.ckpt'
checkpoint = torch.load(checkpoint_path, map_location=torch.device('cpu'))
network.load_state_dict(checkpoint)

# Set network to evaluation mode
network.eval()
