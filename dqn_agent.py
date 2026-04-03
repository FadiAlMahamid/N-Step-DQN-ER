import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import numpy as np
from collections import deque
from q_network import QNetwork
from replay_buffer import ReplayBuffer


# ===================================================================
# --- The N-Step DQN Agent Class ---
# ===================================================================
class DQNAgent:
    """
    The N-Step DQN agent with Experience Replay and a Target Network.
    Uses epsilon-greedy exploration and multi-step returns for faster
    reward propagation.

    N-Step returns replace the standard single-step TD target with a
    multi-step discounted return that accumulates rewards over N steps
    before bootstrapping from the Q-network:

        Single-step: target = rₜ + γ · Q(sₜ₊₁, a')
        N-step:      target = rₜ + γ · rₜ₊₁ + ... + γⁿ⁻¹ · rₜ₊ₙ₋₁
                              + γⁿ · Q(sₜ₊ₙ, a')

    This reduces the bias of the bootstrapped estimate (less reliance on
    potentially inaccurate Q-values) at the cost of slightly higher variance.

    Combined with Double DQN target calculation to reduce overestimation:
        target = Rₙ + γⁿ · Q_target(sₜ₊ₙ, argmaxₐ' Q_policy(sₜ₊ₙ, a'))
    """
    def __init__(self, state_shape, action_size, learning_rate, gamma,
                 device, buffer_capacity, batch_size, learning_starts,
                 target_update_freq, n_steps=3,
                 epsilon_start=1.0, epsilon_min=0.01, epsilon_decay_steps=1000000,
                 loss_function="huber", optimizer="adam",
                 grad_clip_norm=1.0, clip_rewards=True):
        """
        Initializes the N-Step DQN Agent.

        Args:
            state_shape (tuple): Shape of the input state (e.g., (4, 84, 84)).
            action_size (int): Number of possible actions.
            learning_rate (float): Learning rate for the optimizer.
            gamma (float): Discount factor for future rewards.
            device (torch.device): Device to run computations on (CPU/GPU/MPS).
            buffer_capacity (int): Maximum number of transitions in the replay buffer.
            batch_size (int): Number of transitions per training mini-batch.
            learning_starts (int): Warmup transitions before learning begins.
            target_update_freq (int): Steps between target network weight copies.
            n_steps (int): Number of steps for multi-step returns.
            epsilon_start (float): Initial exploration rate.
            epsilon_min (float): Minimum exploration rate after decay.
            epsilon_decay_steps (int): Steps over which epsilon decays linearly.
            loss_function (str): Loss type ("huber" or "mse").
            optimizer (str): Optimizer type ("adam" or "rmsprop").
            grad_clip_norm (float): Maximum gradient norm for clipping.
            clip_rewards (bool): Whether to clip rewards to [-1, 1].
        """
        self.state_shape = state_shape
        self.action_size = action_size
        self.gamma = gamma
        self.device = device
        self.batch_size = batch_size
        self.learning_starts = learning_starts
        self.target_update_freq = target_update_freq
        self.grad_clip_norm = grad_clip_norm
        self.clip_rewards = clip_rewards

        # N-step configuration
        self.n_steps = n_steps
        # Local buffer to accumulate the most recent N transitions before
        # computing the N-step return and storing in the main replay buffer.
        # maxlen=n_steps ensures the oldest transition is automatically
        # discarded when a new one is appended beyond capacity.
        self.n_step_buffer = deque(maxlen=n_steps)

        # Epsilon-greedy exploration parameters
        self.epsilon = epsilon_start
        self.epsilon_start = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay_steps = epsilon_decay_steps

        # Total steps taken across all training episodes
        self.total_steps = 0

        # Counter for tracking when to update the target network
        self.learn_step_counter = 0

        # Initialize the Policy Network (the one we actively train).
        self.policy_network = QNetwork(state_shape, action_size).to(self.device)

        # Initialize the Target Network (a frozen copy used for stable TD targets).
        # The target network is NOT trained directly — its weights are periodically
        # copied from the policy network to provide stable Q-value targets.
        self.target_network = QNetwork(state_shape, action_size).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()

        # Initialize the Experience Replay Buffer.
        # Stores individual frames and reconstructs stacked states on demand
        # to avoid storing redundant overlapping frames.
        frame_shape = state_shape[1:]  # (H, W) from (stack_size, H, W)
        stack_size = state_shape[0]
        self.replay_buffer = ReplayBuffer(buffer_capacity, frame_shape=frame_shape, stack_size=stack_size)

        # Set up the optimizer to update the policy network weights
        if optimizer == "rmsprop":
            self.optimizer = optim.RMSprop(self.policy_network.parameters(), lr=learning_rate)
        else:
            self.optimizer = optim.Adam(self.policy_network.parameters(), lr=learning_rate)

        # Set up the loss function.
        if loss_function == "huber":
            self.loss_fn = nn.SmoothL1Loss()
        else:
            self.loss_fn = nn.MSELoss()

    def choose_action(self, state, use_epsilon=True):
        """
        Chooses an action using epsilon-greedy policy.

        With probability epsilon, a random action is chosen (exploration).
        Otherwise, the action with the highest Q-value is selected (exploitation).
        Epsilon decays linearly over training to shift from exploration to exploitation.

        Args:
            state (np.ndarray): The current state observation.
            use_epsilon (bool): If True (training mode), apply epsilon-greedy
                exploration and decay epsilon. If False (deployment mode),
                always select the greedy action without exploration.

        Returns:
            int: The selected action index.
        """
        if use_epsilon:
            # Decay epsilon before deciding, so the exploration rate
            # reflects the current step count. Then increment the step counter.
            self.update_epsilon()
            self.total_steps += 1

        # Epsilon-greedy: with probability epsilon pick a random action,
        # otherwise pick the action with the highest predicted Q-value.
        if use_epsilon and np.random.rand() < self.epsilon:
            # EXPLORE: Choose a random action to discover new strategies
            return np.random.randint(self.action_size)
        else:
            # EXPLOIT: Choose the action with the highest Q-value.
            # torch.no_grad() disables gradient tracking since we're only
            # doing inference here, not training — saves memory and compute.
            with torch.no_grad():
                # unsqueeze(0) adds a batch dimension: (4,84,84) -> (1,4,84,84)
                state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)

                # Forward pass returns Q-values for all actions, shape: (1, num_actions)
                q_values = self.policy_network(state_tensor)

                # argmax returns the action index with the highest Q-value
                return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        """
        Accumulates transitions in the N-step buffer and stores completed
        N-step transitions in the main replay buffer.

        Instead of storing the raw single-step transition directly, we
        accumulate N transitions and compute the discounted N-step return:
            Rₙ = rₜ + γ · rₜ₊₁ + ... + γⁿ⁻¹ · rₜ₊ₙ₋₁

        The stored transition becomes: (sₜ, aₜ, Rₙ, sₜ₊ₙ, doneₜ₊ₙ)

        This allows the agent to learn from multi-step returns, which
        propagate rewards back to earlier states faster than single-step TD.

        Args:
            state (np.ndarray): The current state.
            action (int): The action taken.
            reward (float): The reward received.
            next_state (np.ndarray): The next state.
            done (bool): Whether the episode terminated.
        """
        # Reward clipping normalizes the learning signal so that games with
        # large score differences train at similar scales.
        if self.clip_rewards:
            reward = np.clip(reward, -1.0, 1.0)

        # 1. Append the current transition to the local N-step buffer
        self.n_step_buffer.append((state, action, reward, next_state, done))

        # 2. Wait until we have accumulated N transitions.
        #    If the episode ends before the buffer is full (e.g., episode
        #    lasts only 2 steps with n_steps=3), flush the partial buffer
        #    so those transitions are not lost.
        if len(self.n_step_buffer) < self.n_steps:
            if done:
                self.flush_n_step_buffer()
            return

        # 3. Calculate the N-step discounted return
        #    Rₙ = rₜ + γ · rₜ₊₁ + γ² · rₜ₊₂ + ... + γⁿ⁻¹ · rₜ₊ₙ₋₁
        n_step_return = sum(
            self.n_step_buffer[i][2] * (self.gamma ** i)
            for i in range(self.n_steps)
        )

        # 4. Extract the initial state/action (at time t) and the final
        #    next_state/done (at time t+n) for the N-step transition
        state_0 = self.n_step_buffer[0][0]       # s_t
        action_0 = self.n_step_buffer[0][1]       # a_t
        next_state_n = self.n_step_buffer[-1][3]  # s_{t+n}
        done_n = self.n_step_buffer[-1][4]        # done_{t+n}

        # 5. Store the N-step transition in the main replay buffer
        self.replay_buffer.push(state_0, action_0, n_step_return, next_state_n, done_n)

        # 6. If the episode ended, flush remaining partial N-step transitions
        #    from the buffer. Each remaining transition gets a shorter-than-N
        #    return (no bootstrapping beyond the terminal state).
        if done:
            self.flush_n_step_buffer()

    def flush_n_step_buffer(self):
        """
        Flushes remaining transitions from the N-step buffer at episode end.

        When an episode terminates, there may be fewer than N transitions
        remaining in the buffer. These are stored with shorter multi-step
        returns (e.g., 2-step, 1-step) so no experience is wasted.

        For example, with N=3 and 2 transitions remaining [t1, t2]:
            - t1 gets a 2-step return: r₁ + γ · r₂
            - t2 gets a 1-step return: r₂
        """
        while len(self.n_step_buffer) > 0:
            # Calculate the return for however many steps remain
            remaining = len(self.n_step_buffer)
            n_step_return = sum(
                self.n_step_buffer[i][2] * (self.gamma ** i)
                for i in range(remaining)
            )

            state_0 = self.n_step_buffer[0][0]
            action_0 = self.n_step_buffer[0][1]
            next_state_n = self.n_step_buffer[-1][3]
            done_n = self.n_step_buffer[-1][4]

            self.replay_buffer.push(state_0, action_0, n_step_return, next_state_n, done_n)

            # Remove the oldest transition and repeat for the remaining ones
            self.n_step_buffer.popleft()

    def learn(self):
        """
        Performs a gradient descent step using a mini-batch from the replay buffer.
        Uses Double DQN target calculation with N-step returns:

            target = Rₙ + γⁿ · Q_target(sₜ₊ₙ, argmaxₐ' Q_policy(sₜ₊ₙ, a'))

        The POLICY network selects the best action (argmax), but the TARGET
        network evaluates that action's Q-value. This decoupling reduces
        overestimation bias. The γⁿ exponent accounts for the fact that
        R_n already covers N steps of discounted rewards.

        Returns:
            float: The loss value for this gradient step.
        """
        # 1. Sample a random mini-batch from the replay buffer
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        # 2. Convert numpy arrays to PyTorch tensors and move to GPU/device.
        #    unsqueeze(1) adds a column dimension so shapes align for element-wise
        #    operations: actions (32,) -> (32,1), rewards (32,) -> (32,1), etc.
        states_t = torch.tensor(states, dtype=torch.float32).to(self.device)              # (batch, 4, 84, 84)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1).to(self.device)  # (batch, 1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1).to(self.device)  # (batch, 1)
        next_states_t = torch.tensor(next_states, dtype=torch.float32).to(self.device)     # (batch, 4, 84, 84)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1).to(self.device)    # (batch, 1)

        # 3. Calculate Q(s, a) - The "Prediction"
        #    policy_network(states_t) returns Q-values for ALL actions: (batch, num_actions)
        #    .gather(1, actions_t) selects only the Q-value for the action that was
        #    actually taken in each transition, resulting in shape (batch, 1).
        q_prediction = self.policy_network(states_t).gather(1, actions_t)

        # 4. Calculate the TD Target using Double DQN with N-step returns
        #
        #   Standard DQN:    target = rₜ + γ · maxₐ' Q_target(sₜ₊₁, a')
        #   Double DQN:      target = rₜ + γ · Q_target(sₜ₊₁, argmaxₐ' Q_policy(sₜ₊₁, a'))
        #   N-Step Double:   target = Rₙ + γⁿ · Q_target(sₜ₊ₙ, argmaxₐ' Q_policy(sₜ₊ₙ, a'))
        #
        # γⁿ is used because the reward Rₙ already accounts for N steps
        # of discounted rewards, so we only need to discount the bootstrapped
        # Q-value by the remaining N steps.
        with torch.no_grad():
            # Step A: Policy network selects the best action for each next state.
            best_actions = self.policy_network(next_states_t).argmax(1, keepdim=True)

            # Step B: Target network evaluates the Q-value of those selected actions.
            q_next = self.target_network(next_states_t).gather(1, best_actions)

            # N-step Bellman Equation: Rₙ + γⁿ · Q_target(sₜ₊ₙ, best_action)
            # (1 - dones_t) zeroes out future reward for terminal states.
            gamma_n = self.gamma ** self.n_steps
            q_target = rewards_t + (1 - dones_t) * gamma_n * q_next

        # 5. Calculate the Loss
        loss = self.loss_fn(q_prediction, q_target)

        # 6. Perform Gradient Descent
        self.optimizer.zero_grad()  # Clear gradients from the previous step
        loss.backward()             # Backpropagate the loss to compute gradients
        # Clip gradients to prevent exploding gradients from large TD errors
        torch.nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.grad_clip_norm)
        self.optimizer.step()       # Update weights using the computed gradients

        # 7. Periodically update the target network
        self.learn_step_counter += 1
        if self.learn_step_counter % self.target_update_freq == 0:
            self.update_target_network()

        return loss.item()

    def update_epsilon(self):
        """
        Performs linear decay on the epsilon value based on total steps taken.
        Transitions the agent from exploration (random) to exploitation (greedy).
        """
        self.epsilon = max(
            self.epsilon_min,
            self.epsilon_start - (self.total_steps * (self.epsilon_start - self.epsilon_min) / self.epsilon_decay_steps)
        )

    def update_target_network(self):
        """
        Hard update: copies policy network weights to the target network.
        Called periodically (every target_update_freq learning steps) to keep
        the target network's Q-values stable between updates.
        """
        self.target_network.load_state_dict(self.policy_network.state_dict())

    def save_model(self, filepath):
        """Saves a full training checkpoint to file."""
        torch.save({
            'policy_network': self.policy_network.state_dict(),
            'target_network': self.target_network.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'learn_step_counter': self.learn_step_counter,
        }, filepath)

    def load_model(self, filepath):
        """Loads a training checkpoint from file into agent components."""
        # map_location ensures tensors are loaded onto the correct device (CPU/GPU/MPS).
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)

        # Support both new checkpoint format (dict with keys) and legacy format
        # (raw state_dict) for backward compatibility with older saved models.
        if isinstance(checkpoint, dict) and 'policy_network' in checkpoint:
            self.policy_network.load_state_dict(checkpoint['policy_network'])
            self.target_network.load_state_dict(checkpoint['target_network'])
            self.optimizer.load_state_dict(checkpoint['optimizer'])
            self.learn_step_counter = checkpoint['learn_step_counter']
        else:
            # Legacy format: checkpoint is just the policy network state_dict
            self.policy_network.load_state_dict(checkpoint)
            self.target_network.load_state_dict(self.policy_network.state_dict())

        self.target_network.eval()
        self.policy_network.train()
