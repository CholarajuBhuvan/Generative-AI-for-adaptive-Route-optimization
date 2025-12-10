"""
Simple RL Agent for route optimization without heavy dependencies
Uses Q-learning approximation and reward-based route selection
"""

import numpy as np
import math
import random
from typing import List, Dict, Tuple, Optional
import json
import logging
from collections import defaultdict, deque


class SimpleRLAgent:
    """
    Simplified RL agent for route optimization
    Uses Q-learning principles without neural networks
    """
    
    def __init__(self, state_dim: int = 32, action_dim: int = 8, 
                 learning_rate: float = 0.1, gamma: float = 0.9, 
                 epsilon: float = 0.1):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.learning_rate = learning_rate
        self.gamma = gamma  # Discount factor
        self.epsilon = epsilon  # Exploration rate
        
        # Q-table approximation using dictionary
        self.q_table = defaultdict(lambda: np.zeros(action_dim))
        self.experience_replay = deque(maxlen=10000)
        self.route_cache = {}
        self.is_initialized = False
        
        # Define action space (route adjustments)
        self.actions = [
            'straight',      # Continue straight
            'slight_left',   # Slight left turn
            'slight_right',  # Slight right turn
            'avoid_traffic', # Avoid high traffic areas
            'shortest',      # Take shortest path
            'fastest',       # Take fastest path
            'scenic',        # Take scenic route
            'main_roads'     # Prefer main roads
        ]
        
    def initialize(self):
        """Initialize the RL agent"""
        try:
            # Initialize Q-values with small random values
            self.q_table.clear()
            
            # Pre-populate some common state-action pairs with reasonable values
            self._initialize_baseline_policy()
            
            self.is_initialized = True
            logging.info("Simple RL Agent initialized successfully")
        except Exception as e:
            logging.error(f"Failed to initialize RL agent: {e}")
            raise
    
    def _initialize_baseline_policy(self):
        """Initialize baseline policy with reasonable default Q-values"""
        # Common traffic scenarios and their preferred actions
        baseline_policies = {
            'high_traffic': {'avoid_traffic': 0.8, 'main_roads': 0.2},
            'low_traffic': {'fastest': 0.7, 'shortest': 0.3},
            'rush_hour': {'avoid_traffic': 0.9, 'main_roads': 0.1},
            'normal_hour': {'shortest': 0.6, 'fastest': 0.4},
            'long_distance': {'fastest': 0.8, 'main_roads': 0.2},
            'short_distance': {'shortest': 0.9, 'straight': 0.1}
        }
        
        for scenario, action_prefs in baseline_policies.items():
            state_key = self._scenario_to_state_key(scenario)
            q_values = np.random.normal(0, 0.1, self.action_dim)
            
            for action_name, value in action_prefs.items():
                if action_name in self.actions:
                    action_idx = self.actions.index(action_name)
                    q_values[action_idx] = value
            
            self.q_table[state_key] = q_values
    
    def encode_state(self, start_point: Tuple[float, float], 
                     end_point: Tuple[float, float], 
                     traffic_data: Dict,
                     current_position: Optional[Tuple[float, float]] = None) -> str:
        """Encode route state into a discrete state key"""
        
        if current_position is None:
            current_position = start_point
        
        # Calculate basic features
        distance = self._calculate_distance(start_point, end_point)
        bearing = self._calculate_bearing(current_position, end_point)
        traffic_level = traffic_data.get('traffic_level', 0.5)
        
        # Discretize continuous features
        distance_bucket = min(int(distance / 5), 20)  # 5km buckets, max 20
        bearing_bucket = int(bearing / 45)  # 8 compass directions
        traffic_bucket = int(traffic_level * 10)  # 0-10 traffic levels
        
        # Time-based features
        import datetime
        now = datetime.datetime.now()
        hour_bucket = now.hour // 6  # 4 time periods per day
        day_bucket = 1 if now.weekday() < 5 else 0  # Weekday vs Weekend
        
        # Create state key
        state_key = f"{distance_bucket}_{bearing_bucket}_{traffic_bucket}_{hour_bucket}_{day_bucket}"
        return state_key
    
    def select_action(self, state_key: str) -> int:
        """Select action using epsilon-greedy policy"""
        
        if random.random() < self.epsilon:
            # Exploration: random action
            return random.randint(0, self.action_dim - 1)
        else:
            # Exploitation: best known action
            q_values = self.q_table[state_key]
            return np.argmax(q_values)
    
    def calculate_reward(self, route_segment: Dict, traffic_data: Dict, 
                        user_preferences: Dict = None) -> float:
        """Calculate reward for a route segment"""
        
        reward = 0.0
        
        # Distance efficiency reward
        segment_distance = route_segment.get('distance', 0)
        if segment_distance > 0:
            reward -= segment_distance * 0.1  # Penalty for longer routes
        
        # Time efficiency reward
        segment_time = route_segment.get('time', 0)
        if segment_time > 0:
            reward -= segment_time * 0.2  # Penalty for slower routes
        
        # Traffic reward
        traffic_level = traffic_data.get('traffic_level', 0.5)
        if traffic_level < 0.3:
            reward += 0.5  # Bonus for avoiding traffic
        elif traffic_level > 0.7:
            reward -= 0.5  # Penalty for high traffic
        
        # Safety reward (prefer main roads)
        if route_segment.get('road_type') == 'highway':
            reward += 0.2
        elif route_segment.get('road_type') == 'residential':
            reward -= 0.1
        
        # User preference rewards
        if user_preferences:
            if user_preferences.get('prefer_highways', False) and \
               route_segment.get('road_type') == 'highway':
                reward += 0.3
            
            if user_preferences.get('avoid_tolls', False) and \
               route_segment.get('has_tolls', False):
                reward -= 0.4
            
            if user_preferences.get('scenic_route', False):
                reward += route_segment.get('scenic_value', 0) * 0.2
        
        return reward
    
    def optimize_route(self, start_point: Tuple[float, float], 
                      end_point: Tuple[float, float], 
                      traffic_data: Dict, 
                      preferences: Dict = None) -> Dict:
        """Generate optimized route using RL approach"""
        
        if not self.is_initialized:
            self.initialize()
        
        # Generate cache key
        cache_key = f"rl_{start_point}_{end_point}_{hash(str(traffic_data))}"
        if cache_key in self.route_cache:
            return self.route_cache[cache_key]
        
        # Generate route using RL decision making
        route_segments = self._generate_rl_route(start_point, end_point, traffic_data, preferences)
        
        # Calculate total metrics
        total_distance = sum(seg.get('distance', 0) for seg in route_segments)
        total_time = sum(seg.get('time', 0) for seg in route_segments)
        total_reward = sum(seg.get('reward', 0) for seg in route_segments)
        
        # Convert segments to waypoints
        waypoints = self._segments_to_waypoints(route_segments, start_point, end_point)
        
        result = {
            'coordinates': waypoints,
            'total_distance_km': total_distance,
            'total_time_minutes': total_time,
            'confidence_score': min(0.95, 0.7 + (total_reward / len(route_segments)) * 0.25),
            'algorithm': 'simple_rl_agent',
            'rl_metrics': {
                'total_reward': total_reward,
                'avg_reward_per_segment': total_reward / len(route_segments),
                'exploration_rate': self.epsilon,
                'q_table_size': len(self.q_table)
            }
        }
        
        # Cache result
        self.route_cache[cache_key] = result
        return result
    
    def _generate_rl_route(self, start_point: Tuple[float, float], 
                          end_point: Tuple[float, float], 
                          traffic_data: Dict,
                          preferences: Dict = None) -> List[Dict]:
        """Generate route segments using RL decision making"""
        
        segments = []
        current_position = start_point
        num_steps = 15  # Increased to 15 for better road snapping coverage
        
        for step in range(num_steps):
            # Encode current state
            state_key = self.encode_state(start_point, end_point, traffic_data, current_position)
            
            # Select action
            action_idx = self.select_action(state_key)
            action_name = self.actions[action_idx]
            
            # Calculate next position based on action
            next_position = self._apply_action(current_position, end_point, action_name, traffic_data)
            
            # Calculate segment metrics
            segment_distance = self._calculate_distance(current_position, next_position)
            segment_time = self._estimate_segment_time(segment_distance, action_name, traffic_data)
            
            segment = {
                'start': current_position,
                'end': next_position,
                'distance': segment_distance,
                'time': segment_time,
                'action': action_name,
                'road_type': self._infer_road_type(action_name),
                'has_tolls': action_name in ['fastest', 'main_roads'],
                'scenic_value': 0.8 if action_name == 'scenic' else 0.3
            }
            
            # Calculate reward
            segment['reward'] = self.calculate_reward(segment, traffic_data, preferences)
            
            segments.append(segment)
            
            # Update current position
            current_position = next_position
            
            # Update Q-value (simplified Q-learning)
            if step > 0:  # Need previous state for Q-learning
                prev_state = segments[step-1]['state_key'] if step > 0 else state_key
                self._update_q_value(prev_state, action_idx, segment['reward'], state_key)
            
            segment['state_key'] = state_key
        
        return segments
    
    def _apply_action(self, current_pos: Tuple[float, float], 
                     target_pos: Tuple[float, float], 
                     action: str, 
                     traffic_data: Dict) -> Tuple[float, float]:
        """Apply action to determine next position"""
        
        # Base progress toward target
        progress = 1.0 / 15  # Move 1/15 of the way to target (matches num_steps)
        base_lat = current_pos[0] + progress * (target_pos[0] - current_pos[0])
        base_lng = current_pos[1] + progress * (target_pos[1] - current_pos[1])
        
        # Apply action-specific adjustments
        lat_offset = 0.0
        lng_offset = 0.0
        
        if action == 'slight_left':
            lat_offset = -0.001
        elif action == 'slight_right':
            lat_offset = 0.001
        elif action == 'avoid_traffic':
            # Move away from high traffic areas
            if traffic_data.get('traffic_level', 0.5) > 0.6:
                lat_offset = random.uniform(-0.002, 0.002)
                lng_offset = random.uniform(-0.002, 0.002)
        elif action == 'scenic':
            # Add slight deviation for scenic routes
            lat_offset = random.uniform(-0.0015, 0.0015)
            lng_offset = random.uniform(-0.0015, 0.0015)
        elif action == 'shortest':
            # More direct path (less offset)
            progress *= 1.1
            base_lat = current_pos[0] + progress * (target_pos[0] - current_pos[0])
            base_lng = current_pos[1] + progress * (target_pos[1] - current_pos[1])
        
        return (base_lat + lat_offset, base_lng + lng_offset)
    
    def _update_q_value(self, state_key: str, action_idx: int, reward: float, next_state_key: str):
        """Update Q-value using Q-learning update rule"""
        
        current_q = self.q_table[state_key][action_idx]
        next_max_q = np.max(self.q_table[next_state_key])
        
        # Q-learning update: Q(s,a) = Q(s,a) + α[r + γ*max(Q(s',a')) - Q(s,a)]
        new_q = current_q + self.learning_rate * (reward + self.gamma * next_max_q - current_q)
        self.q_table[state_key][action_idx] = new_q
    
    def _segments_to_waypoints(self, segments: List[Dict], 
                              start_point: Tuple[float, float], 
                              end_point: Tuple[float, float]) -> List[Dict]:
        """Convert route segments to waypoint format"""
        
        waypoints = []
        cumulative_time = 0.0
        
        # Add start point
        waypoints.append({
            'lat': start_point[0],
            'lng': start_point[1],
            'time': 0.0,
            'confidence': 0.95,
            'action': 'start'
        })
        
        # Add intermediate waypoints
        for i, segment in enumerate(segments):
            cumulative_time += segment['time']
            waypoints.append({
                'lat': segment['end'][0],
                'lng': segment['end'][1],
                'time': cumulative_time,
                'confidence': 0.85 + segment['reward'] * 0.1,
                'action': segment['action'],
                'reward': segment['reward']
            })
        
        # Ensure end point is correct
        waypoints[-1]['lat'] = end_point[0]
        waypoints[-1]['lng'] = end_point[1]
        waypoints[-1]['action'] = 'destination'
        
        return waypoints
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points using Haversine formula"""
        R = 6371  # Earth's radius in km
        
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.asin(math.sqrt(a))
        
        return R * c
    
    def _calculate_bearing(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate bearing between two points"""
        lat1, lon1 = math.radians(point1[0]), math.radians(point1[1])
        lat2, lon2 = math.radians(point2[0]), math.radians(point2[1])
        
        dlon = lon2 - lon1
        y = math.sin(dlon) * math.cos(lat2)
        x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
        
        bearing = math.atan2(y, x)
        return (math.degrees(bearing) + 360) % 360
    
    def _estimate_segment_time(self, distance: float, action: str, traffic_data: Dict) -> float:
        """Estimate time for a route segment"""
        # Realistic base speed based on distance (km/h)
        if distance < 10:
            base_speed = 35  # Urban driving
        elif distance < 50:
            base_speed = 50  # Mixed roads
        else:
            base_speed = 60  # Highway
        
        # Adjust speed based on action
        speed_multipliers = {
            'fastest': 1.15,      # Highways, faster routes
            'main_roads': 1.10,   # Main arterial roads
            'shortest': 1.0,      # Direct route
            'straight': 1.0,      # Straight path
            'avoid_traffic': 0.85, # Alternative routes
            'scenic': 0.75,       # Scenic/slower routes
            'slight_left': 0.95,  # Minor turns
            'slight_right': 0.95  # Minor turns
        }
        
        speed = base_speed * speed_multipliers.get(action, 1.0)
        
        # Adjust for traffic
        traffic_factor = traffic_data.get('traffic_level', 0.5)
        speed *= (1 - traffic_factor * 0.5)
        
        # Ensure minimum realistic speed
        speed = max(speed, 15.0)
        
        # Calculate time with 8% delay overhead
        return (distance / speed) * 60 * 1.08  # Convert to minutes with delays
    
    def _infer_road_type(self, action: str) -> str:
        """Infer road type from action"""
        road_type_map = {
            'fastest': 'highway',
            'main_roads': 'highway',
            'shortest': 'arterial',
            'straight': 'arterial',
            'avoid_traffic': 'local',
            'scenic': 'scenic',
            'slight_left': 'arterial',
            'slight_right': 'arterial'
        }
        return road_type_map.get(action, 'arterial')
    
    def _scenario_to_state_key(self, scenario: str) -> str:
        """Convert scenario name to state key format"""
        scenario_mappings = {
            'high_traffic': "5_0_8_2_1",
            'low_traffic': "5_0_2_2_1", 
            'rush_hour': "10_0_9_1_1",
            'normal_hour': "10_0_4_2_1",
            'long_distance': "15_0_4_2_1",
            'short_distance': "2_0_4_2_1"
        }
        return scenario_mappings.get(scenario, "5_0_4_2_1")
    
    def get_model_info(self) -> Dict:
        """Get information about the RL agent"""
        return {
            'model_type': 'SimpleRLAgent',
            'state_dim': self.state_dim,
            'action_dim': self.action_dim,
            'actions': self.actions,
            'learning_rate': self.learning_rate,
            'gamma': self.gamma,
            'epsilon': self.epsilon,
            'is_initialized': self.is_initialized,
            'q_table_size': len(self.q_table),
            'cache_size': len(self.route_cache)
        }
