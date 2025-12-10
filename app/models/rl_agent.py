"""
Reinforcement Learning Agent for Adaptive Route Optimization
Uses Deep Q-Network (DQN) to learn optimal routing strategies
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque
from typing import List, Dict, Tuple, Optional
import json
from pathlib import Path


class RouteEnvironment:
    """
    Environment for reinforcement learning route optimization
    """
    
    def __init__(self, map_data: Dict, traffic_data: Dict):
        self.map_data = map_data
        self.traffic_data = traffic_data
        self.current_position = None
        self.target_position = None
        self.route_history = []
        self.step_count = 0
        self.max_steps = 100
        
    def reset(self, start: Tuple[float, float], end: Tuple[float, float]) -> np.ndarray:
        """Reset environment with new start and end points"""
        self.current_position = start
        self.target_position = end
        self.route_history = [start]
        self.step_count = 0
        return self._get_state()
    
    def step(self, action: int) -> Tuple[np.ndarray, float, bool, Dict]:
        """
        Execute action and return new state, reward, done, info
        
        Args:
            action: Action to take (0-7 representing 8 directions)
            
        Returns:
            Tuple of (new_state, reward, done, info)
        """
        # Convert action to direction
        directions = [
            (0.001, 0), (0.001, 0.001), (0, 0.001), (-0.001, 0.001),
            (-0.001, 0), (-0.001, -0.001), (0, -0.001), (0.001, -0.001)
        ]
        
        lat_delta, lon_delta = directions[action]
        
        # Update position
        new_lat = self.current_position[0] + lat_delta
        new_lon = self.current_position[1] + lon_delta
        new_position = (new_lat, new_lon)
        
        # Update route history
        self.route_history.append(new_position)
        self.current_position = new_position
        self.step_count += 1
        
        # Calculate reward
        reward = self._calculate_reward()
        
        # Check if done
        done = self._is_done()
        
        # Get info
        info = {
            'position': new_position,
            'distance_to_target': self._distance_to_target(),
            'steps': self.step_count
        }
        
        return self._get_state(), reward, done, info
    
    def _get_state(self) -> np.ndarray:
        """Get current state representation"""
        # Normalize positions
        current_norm = [self.current_position[0] / 90.0, self.current_position[1] / 180.0]
        target_norm = [self.target_position[0] / 90.0, self.target_position[1] / 180.0]
        
        # Get traffic information at current position
        traffic_level = self._get_traffic_at_position(self.current_position)
        
        # Get distance to target
        distance_to_target = self._distance_to_target() / 100.0  # Normalize
        
        # Combine all features
        state = current_norm + target_norm + [traffic_level, distance_to_target]
        
        # Pad to fixed size
        while len(state) < 32:
            state.append(0.0)
        
        return np.array(state, dtype=np.float32)
    
    def _calculate_reward(self) -> float:
        """Calculate reward for current state"""
        # Distance reward (closer to target is better)
        distance = self._distance_to_target()
        distance_reward = max(0, 1.0 - distance / 10.0)
        
        # Traffic penalty
        traffic_level = self._get_traffic_at_position(self.current_position)
        traffic_penalty = -traffic_level * 0.1
        
        # Step penalty (encourage efficiency)
        step_penalty = -0.01
        
        # Success bonus
        success_bonus = 10.0 if distance < 0.01 else 0.0
        
        # Collision penalty
        collision_penalty = -5.0 if self._is_collision() else 0.0
        
        return distance_reward + traffic_penalty + step_penalty + success_bonus + collision_penalty
    
    def _distance_to_target(self) -> float:
        """Calculate Euclidean distance to target"""
        lat_diff = self.current_position[0] - self.target_position[0]
        lon_diff = self.current_position[1] - self.target_position[1]
        return np.sqrt(lat_diff**2 + lon_diff**2)
    
    def _get_traffic_at_position(self, position: Tuple[float, float]) -> float:
        """Get traffic level at given position"""
        # Simplified traffic model - in production, use real traffic data
        lat, lon = position
        # Generate synthetic traffic based on position
        traffic = 0.3 + 0.4 * np.sin(lat * 10) * np.cos(lon * 10)
        return max(0, min(1, traffic))
    
    def _is_collision(self) -> bool:
        """Check if current position is a collision (obstacle)"""
        # Simplified collision detection
        lat, lon = self.current_position
        
        # Check for water bodies (simplified)
        if abs(lat) > 0.5 or abs(lon) > 0.5:
            return True
        
        # Check for restricted areas
        if 0.1 < lat < 0.2 and 0.1 < lon < 0.2:
            return True
        
        return False
    
    def _is_done(self) -> bool:
        """Check if episode is done"""
        return (self._distance_to_target() < 0.01 or 
                self.step_count >= self.max_steps or 
                self._is_collision())


class DQN(nn.Module):
    """
    Deep Q-Network for route optimization
    """
    
    def __init__(self, state_dim: int = 32, action_dim: int = 8, hidden_dim: int = 256):
        super().__init__()
        
        self.state_dim = state_dim
        self.action_dim = action_dim
        
        # Network architecture
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
    
    def _init_weights(self, module):
        """Initialize network weights"""
        if isinstance(module, nn.Linear):
            torch.nn.init.xavier_uniform_(module.weight)
            module.bias.data.fill_(0.01)
    
    def forward(self, state: torch.Tensor) -> torch.Tensor:
        """Forward pass"""
        return self.network(state)
    
    def act(self, state: torch.Tensor, epsilon: float = 0.0) -> int:
        """
        Choose action using epsilon-greedy policy
        
        Args:
            state: Current state
            epsilon: Exploration rate
            
        Returns:
            Selected action
        """
        if random.random() < epsilon:
            return random.randrange(self.action_dim)
        
        with torch.no_grad():
            q_values = self.forward(state)
            return q_values.argmax().item()


class ReplayBuffer:
    """Experience replay buffer for DQN"""
    
    def __init__(self, capacity: int = 10000):
        self.buffer = deque(maxlen=capacity)
    
    def push(self, state, action, reward, next_state, done):
        """Add experience to buffer"""
        self.buffer.append((state, action, reward, next_state, done))
    
    def sample(self, batch_size: int) -> List:
        """Sample batch of experiences"""
        return random.sample(self.buffer, batch_size)
    
    def __len__(self):
        return len(self.buffer)


class RLAgent:
    """
    Reinforcement Learning Agent for route optimization
    """
    
    def __init__(self, 
                 state_dim: int = 32,
                 action_dim: int = 8,
                 learning_rate: float = 1e-4,
                 gamma: float = 0.99,
                 epsilon_start: float = 1.0,
                 epsilon_end: float = 0.01,
                 epsilon_decay: int = 10000):
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Networks
        self.q_network = DQN(state_dim, action_dim).to(self.device)
        self.target_network = DQN(state_dim, action_dim).to(self.device)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        
        # Hyperparameters
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        
        # Experience replay
        self.replay_buffer = ReplayBuffer()
        self.batch_size = 32
        
        # Update counter
        self.update_count = 0
        self.target_update_freq = 1000
        
        # Training history
        self.training_history = {
            'rewards': [],
            'losses': [],
            'epsilon': []
        }
    
    def select_action(self, state: np.ndarray) -> int:
        """Select action using current policy"""
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        action = self.q_network.act(state_tensor, self.epsilon)
        
        # Decay epsilon
        self.epsilon = max(self.epsilon_end, 
                          self.epsilon - (self.epsilon_start - self.epsilon_end) / self.epsilon_decay)
        
        return action
    
    def store_experience(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.replay_buffer.push(state, action, reward, next_state, done)
    
    def train(self) -> Optional[float]:
        """Train the agent on a batch of experiences"""
        if len(self.replay_buffer) < self.batch_size:
            return None
        
        # Sample batch
        batch = self.replay_buffer.sample(self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        
        # Convert to tensors
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.BoolTensor(dones).to(self.device)
        
        # Current Q values
        current_q_values = self.q_network(states).gather(1, actions.unsqueeze(1))
        
        # Next Q values from target network
        with torch.no_grad():
            next_q_values = self.target_network(next_states).max(1)[0]
            target_q_values = rewards + (self.gamma * next_q_values * ~dones)
        
        # Compute loss
        loss = nn.MSELoss()(current_q_values.squeeze(), target_q_values)
        
        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.q_network.parameters(), max_norm=1.0)
        self.optimizer.step()
        
        # Update target network
        self.update_count += 1
        if self.update_count % self.target_update_freq == 0:
            self.target_network.load_state_dict(self.q_network.state_dict())
        
        return loss.item()
    
    def optimize_route(self, 
                      start: Tuple[float, float],
                      end: Tuple[float, float],
                      map_data: Dict,
                      traffic_data: Dict,
                      num_episodes: int = 100) -> Dict:
        """
        Optimize route using RL agent
        
        Args:
            start: Starting coordinates
            end: Ending coordinates
            map_data: Map information
            traffic_data: Traffic information
            num_episodes: Number of training episodes
            
        Returns:
            Optimized route with metadata
        """
        # Create environment
        env = RouteEnvironment(map_data, traffic_data)
        
        episode_rewards = []
        
        for episode in range(num_episodes):
            state = env.reset(start, end)
            episode_reward = 0
            done = False
            
            while not done:
                # Select action
                action = self.select_action(state)
                
                # Execute action
                next_state, reward, done, info = env.step(action)
                
                # Store experience
                self.store_experience(state, action, reward, next_state, done)
                
                # Train
                loss = self.train()
                
                # Update state and reward
                state = next_state
                episode_reward += reward
            
            episode_rewards.append(episode_reward)
            
            # Log progress
            if episode % 10 == 0:
                avg_reward = np.mean(episode_rewards[-10:])
                print(f"Episode {episode}, Average Reward: {avg_reward:.2f}, Epsilon: {self.epsilon:.3f}")
        
        # Generate final route
        final_route = self._generate_final_route(env, start, end)
        
        return {
            'route': final_route,
            'episode_rewards': episode_rewards,
            'final_epsilon': self.epsilon,
            'total_episodes': num_episodes
        }
    
    def _generate_final_route(self, 
                             env: RouteEnvironment,
                             start: Tuple[float, float],
                             end: Tuple[float, float]) -> List[Dict]:
        """Generate final route using learned policy (epsilon=0)"""
        state = env.reset(start, end)
        route = [{'lat': start[0], 'lng': start[1], 'step': 0}]
        done = False
        
        while not done:
            # Use greedy policy (epsilon=0)
            action = self.select_action(state)
            state, reward, done, info = env.step(action)
            
            route.append({
                'lat': info['position'][0],
                'lng': info['position'][1],
                'step': info['steps'],
                'reward': reward
            })
        
        return route
    
    def save_model(self, path: str):
        """Save the trained model"""
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save networks
        torch.save(self.q_network.state_dict(), model_path / "q_network.pt")
        torch.save(self.target_network.state_dict(), model_path / "target_network.pt")
        
        # Save optimizer state
        torch.save(self.optimizer.state_dict(), model_path / "optimizer.pt")
        
        # Save training configuration
        config = {
            'state_dim': 32,
            'action_dim': 8,
            'learning_rate': 1e-4,
            'gamma': self.gamma,
            'epsilon_start': 1.0,
            'epsilon_end': self.epsilon_end,
            'epsilon_decay': self.epsilon_decay,
            'update_count': self.update_count,
            'epsilon': self.epsilon
        }
        
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f)
        
        # Save training history
        with open(model_path / "training_history.json", "w") as f:
            json.dump(self.training_history, f)
    
    @classmethod
    def load_model(cls, path: str):
        """Load a trained model"""
        model_path = Path(path)
        
        # Load configuration
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create agent
        agent = cls(**{k: v for k, v in config.items() if k not in ['update_count', 'epsilon']})
        
        # Load networks
        agent.q_network.load_state_dict(torch.load(model_path / "q_network.pt", map_location=agent.device))
        agent.target_network.load_state_dict(torch.load(model_path / "target_network.pt", map_location=agent.device))
        agent.optimizer.load_state_dict(torch.load(model_path / "optimizer.pt", map_location=agent.device))
        
        # Restore training state
        agent.update_count = config['update_count']
        agent.epsilon = config['epsilon']
        
        # Load training history
        if (model_path / "training_history.json").exists():
            with open(model_path / "training_history.json", "r") as f:
                agent.training_history = json.load(f)
        
        return agent
