import numpy as np
import random


# ===================================================================
# --- The Experience Replay Buffer ---
# ===================================================================
class ReplayBuffer:
    """
    A memory-efficient, fixed-size buffer to store and sample experience tuples.

    Experience Replay breaks the temporal correlation between consecutive
    samples by storing transitions and sampling random mini-batches for
    training. This prevents catastrophic forgetting that occurs when an
    agent only learns from the most recent transition and overwrites
    previously learned behavior.

    Instead of storing full frame-stacked states (which duplicate 3 of 4
    frames between consecutive transitions), this buffer stores individual
    frames and reconstructs stacked states on demand during sampling.
    This reduces memory usage by nearly half compared to a naive approach.

    For N-Step DQN, the transitions stored here already contain the
    N-step discounted return and the state N steps ahead. The N-step
    accumulation happens in the agent before calling push().

    Each transition is stored as: (state, action, reward, next_state, done)
    """

    def __init__(self, capacity, frame_shape=(84, 84), stack_size=4):
        """
        Initializes the replay buffer.

        Args:
            capacity (int): Maximum number of transitions to store.
                When full, the oldest transitions are discarded.
            frame_shape (tuple): Height and width of a single preprocessed frame.
            stack_size (int): Number of frames stacked together to form a state.
        """
        self.capacity = capacity
        self.stack_size = stack_size

        # pos: the index where the NEXT transition will be written (circular)
        self.pos = 0
        # size: how many transitions have been stored so far (capped at capacity)
        self.size = 0

        # Pre-allocate fixed-size arrays for each component of a transition.
        # Using uint8 for frames saves 4x memory vs float32 (1 byte vs 4 bytes per pixel).
        # For 1M capacity with 84x84 frames: ~7 GB (uint8) vs ~28 GB (float32).
        self.frames = np.zeros((capacity, *frame_shape), dtype=np.uint8)
        self.actions = np.zeros(capacity, dtype=np.int64)
        self.rewards = np.zeros(capacity, dtype=np.float32)
        self.dones = np.zeros(capacity, dtype=np.bool_)

    def push(self, state, action, reward, next_state, done):
        """
        Adds a transition to the buffer.

        Only the latest frame from next_state is stored, since the
        preceding frames are already in the buffer from prior transitions.

        Args:
            state (np.ndarray): The current frame-stacked state (stack_size, H, W).
            action (int): The action taken.
            reward (float): The reward received (may be an N-step discounted return).
            next_state (np.ndarray): The next frame-stacked state (stack_size, H, W).
            done (bool): Whether the episode terminated.
        """
        # Store only the newest frame from next_state (the last in the stack).
        # Example: if next_state = [f1, f2, f3, f4], we store only f4.
        # Frames f1-f3 were already stored by previous transitions.
        self.frames[self.pos] = next_state[-1]
        self.actions[self.pos] = action
        self.rewards[self.pos] = reward
        self.dones[self.pos] = done

        # Advance the write pointer (wraps around for circular buffer behavior)
        self.pos = (self.pos + 1) % self.capacity
        self.size = min(self.size + 1, self.capacity)

    def _get_stacked_state(self, idx):
        """
        Reconstructs a frame-stacked state ending at the given index.

        If a frame boundary crosses an episode done flag, earlier frames
        are zeroed out to avoid leaking information from a previous episode.

        Args:
            idx (int): The index of the last frame in the stack.

        Returns:
            np.ndarray: The stacked state of shape (stack_size, H, W).
        """
        # Build the list of frame indices that make up this stacked state.
        # For stack_size=4 and idx=10: indices = [7, 8, 9, 10]
        # Modular arithmetic handles wraparound at the buffer boundary.
        indices = [(idx - i) % self.capacity for i in reversed(range(self.stack_size))]
        stack = self.frames[indices]

        # Zero out frames that belong to a previous episode to prevent
        # the agent from "seeing" the end of one episode blended into the
        # start of another. If done[i] is True, frames at index i and
        # earlier came from a finished episode and must be blanked.
        # We only check up to stack_size-1 because the last frame (the
        # "current" frame) is always valid by definition.
        for i in range(self.stack_size - 1):
            if self.dones[indices[i]]:
                stack[:i + 1] = 0
                break
        return stack

    def sample(self, batch_size):
        """
        Samples a random mini-batch of transitions from the buffer.

        Args:
            batch_size (int): The number of transitions to sample.

        Returns:
            tuple: A tuple of numpy arrays (states, actions, rewards,
                next_states, dones), each with batch_size elements.
        """
        # Sample valid indices, avoiding two edge cases:
        #   1. The current write position (self.pos) — that slot may be partially
        #      overwritten and doesn't represent a complete transition.
        #   2. Indices too close to the start of a non-full buffer, where there
        #      aren't enough preceding frames to reconstruct a full stack.
        indices = []
        while len(indices) < batch_size:
            idx = random.randint(self.stack_size, self.size - 1) if self.size < self.capacity \
                else random.randint(0, self.capacity - 1)
            if idx == self.pos:
                continue
            indices.append(idx)

        # Reconstruct stacked states from individual frames:
        #   - "state" (s)  = stack ending one position BEFORE the stored index,
        #     because the stored frame at `idx` is the newest frame of next_state.
        #   - "next_state" (s') = stack ending AT the stored index.
        states = np.array([self._get_stacked_state((idx - 1) % self.capacity) for idx in indices], dtype=np.float32)
        next_states = np.array([self._get_stacked_state(idx) for idx in indices], dtype=np.float32)
        actions = self.actions[indices]
        rewards = self.rewards[indices]
        # Convert boolean dones to float for use in the Bellman equation: (1 − doneₜ) · γ · Q
        dones = self.dones[indices].astype(np.float32)

        return states, actions, rewards, next_states, dones

    def __len__(self):
        """Returns the current number of transitions stored in the buffer."""
        return self.size
