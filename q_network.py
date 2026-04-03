import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


# ===================================================================
# --- Standard DQN Q-Network ---
# ===================================================================
class QNetwork(nn.Module):
    """
    Convolutional Neural Network (CNN) for Q-value approximation.

    Based on the DeepMind 2015 Nature CNN architecture with standard
    fully connected layers. This is the same network used across
    standard DQN, Double DQN, and N-Step DQN — the N-step modification
    changes the target calculation in the agent, not the network structure.
    """
    def __init__(self, input_shape, num_actions):
        """
        Initializes the network layers.

        Args:
            input_shape (tuple): The shape of the input state (e.g., (4, 84, 84)).
            num_actions (int): The number of possible actions.
        """
        super(QNetwork, self).__init__()

        # Assumes input is (Channels, Height, Width)
        # For Atari, input_shape is (4, 84, 84) -> (FrameStack, H, W)
        in_channels = input_shape[0]

        # Convolutional layers as described in the DeepMind 2015 Nature paper.
        # Each layer progressively reduces spatial dimensions while increasing feature depth:
        #   Input:  (4, 84, 84)  -- 4 stacked grayscale frames
        #   conv1:  (32, 20, 20) -- large 8x8 filters with stride 4 capture coarse features
        #   conv2:  (64, 9, 9)   -- 4x4 filters with stride 2 capture mid-level patterns
        #   conv3:  (64, 7, 7)   -- 3x3 filters with stride 1 capture fine-grained details
        self.conv1 = nn.Conv2d(in_channels, 32, kernel_size=8, stride=4)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(64, 64, kernel_size=3, stride=1)

        # Calculate the size of the flattened feature map after conv layers.
        # A dummy forward pass determines the output size dynamically, so the code
        # doesn't break if input_shape changes (e.g., different screen_size).
        # For (4, 84, 84) input: conv_out_size = 64 * 7 * 7 = 3136.
        dummy_input = torch.zeros(1, *input_shape)
        conv_out_size = self._get_conv_out(dummy_input)

        # Fully connected layers map the flattened conv features to Q-values.
        # fc1 maps the flattened conv features (3136) to a 512-unit hidden layer.
        # fc2 maps the hidden layer to Q-values, one per action (4 for Breakout:
        # NOOP, FIRE, RIGHT, LEFT). No activation on fc2 because Q-values can
        # be any real number (positive or negative).
        self.fc1 = nn.Linear(conv_out_size, 512)
        self.fc2 = nn.Linear(512, num_actions)

    def _get_conv_out(self, x):
        """Helper function to calculate the output size of the conv layers."""
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))
        # Flatten the tensor, except for the batch dimension, to get the total size
        return int(np.prod(x.size()[1:]))

    def forward(self, x):
        """
        Defines the forward pass of the network.

        Args:
            x (torch.Tensor): The input batch of states.

        Returns:
            torch.Tensor: The Q-values for each action.
        """
        # Normalize pixel values from [0, 255] to [0.0, 1.0] for stable training
        # Doing this on the GPU here is faster than doing it on CPU before creating the tensor.
        x = x / 255.0

        # Pass through convolutional layers with ReLU activation
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.relu(self.conv3(x))

        # Flatten the output from the conv layers into a 1D vector
        x = x.view(x.size(0), -1)

        # Pass through the fully connected layers
        x = F.relu(self.fc1(x))
        q_values = self.fc2(x)

        return q_values
