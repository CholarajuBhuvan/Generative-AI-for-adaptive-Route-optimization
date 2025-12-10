"""
Transformer-based route generation model using attention mechanisms
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import List, Dict, Tuple, Optional
import json
import pickle
from pathlib import Path


class PositionalEncoding(nn.Module):
    """Positional encoding for transformer"""
    
    def __init__(self, d_model: int, max_len: int = 5000):
        super().__init__()
        self.dropout = nn.Dropout(p=0.1)
        
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-np.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)
    
    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class RouteTransformer(nn.Module):
    """
    Transformer model for generating optimal routes
    Uses attention mechanisms to understand spatial and temporal patterns
    """
    
    def __init__(self, 
                 input_dim: int = 128,
                 d_model: int = 256,
                 nhead: int = 8,
                 num_layers: int = 6,
                 dim_feedforward: int = 1024,
                 max_route_length: int = 100,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.max_route_length = max_route_length
        
        # Input projection
        self.input_projection = nn.Linear(input_dim, d_model)
        
        # Positional encoding
        self.pos_encoding = PositionalEncoding(d_model, max_route_length)
        
        # Transformer encoder
        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=dim_feedforward,
            dropout=dropout,
            batch_first=True
        )
        self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
        
        # Output heads for different route aspects
        self.route_head = nn.Linear(d_model, 4)  # lat, lon, time, confidence
        self.traffic_head = nn.Linear(d_model, 1)  # traffic prediction
        self.preference_head = nn.Linear(d_model, 5)  # user preferences
        
        # Route validation
        self.route_validator = nn.Sequential(
            nn.Linear(d_model, 64),
            nn.ReLU(),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
    
    def forward(self, route_features: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Forward pass through the transformer
        
        Args:
            route_features: Input features [batch_size, seq_len, input_dim]
            attention_mask: Attention mask for padding
            
        Returns:
            Dictionary with route predictions
        """
        batch_size, seq_len, _ = route_features.shape
        
        # Project input to model dimension
        x = self.input_projection(route_features)
        
        # Add positional encoding
        x = x.transpose(0, 1)  # [seq_len, batch_size, d_model]
        x = self.pos_encoding(x)
        x = x.transpose(0, 1)  # [batch_size, seq_len, d_model]
        
        # Apply attention mask if provided
        if attention_mask is not None:
            # Convert attention mask to transformer format
            attention_mask = (1 - attention_mask) * -1e9
        
        # Transformer encoding
        encoded = self.transformer(x, src_key_padding_mask=attention_mask)
        
        # Generate outputs
        outputs = {
            'route_coordinates': self.route_head(encoded),  # [batch, seq, 4]
            'traffic_prediction': self.traffic_head(encoded),  # [batch, seq, 1]
            'preference_scores': self.preference_head(encoded),  # [batch, seq, 5]
            'route_validity': self.route_validator(encoded)  # [batch, seq, 1]
        }
        
        return outputs
    
    def generate_route(self, 
                      start_point: Tuple[float, float],
                      end_point: Tuple[float, float],
                      constraints: Dict,
                      num_alternatives: int = 5) -> List[Dict]:
        """
        Generate route alternatives using the transformer model
        
        Args:
            start_point: (latitude, longitude) of start
            end_point: (latitude, longitude) of end
            constraints: Route constraints (time, distance, preferences)
            num_alternatives: Number of route alternatives to generate
            
        Returns:
            List of route alternatives with metadata
        """
        self.eval()
        
        with torch.no_grad():
            # Create input features
            input_features = self._create_input_features(
                start_point, end_point, constraints
            )
            
            # Generate multiple routes
            routes = []
            for _ in range(num_alternatives):
                # Forward pass
                outputs = self.forward(input_features)
                
                # Decode route
                route = self._decode_route(outputs, start_point, end_point)
                routes.append(route)
            
            # Sort routes by quality score
            routes.sort(key=lambda x: x['quality_score'], reverse=True)
            
            return routes
    
    def _create_input_features(self, 
                              start_point: Tuple[float, float],
                              end_point: Tuple[float, float],
                              constraints: Dict) -> torch.Tensor:
        """Create input feature tensor from route parameters"""
        
        # Normalize coordinates
        start_norm = self._normalize_coordinates(start_point)
        end_norm = self._normalize_coordinates(end_point)
        
        # Create sequence of waypoints (simplified for demo)
        waypoints = self._interpolate_waypoints(start_norm, end_norm, 50)
        
        # Add constraint features
        constraint_features = [
            constraints.get('max_time', 60) / 180.0,  # Normalized time
            constraints.get('max_distance', 50) / 200.0,  # Normalized distance
            constraints.get('avoid_tolls', 0),
            constraints.get('avoid_highways', 0),
            constraints.get('prefer_scenic', 0)
        ]
        
        # Combine features
        features = []
        for i, waypoint in enumerate(waypoints):
            feature_vector = waypoint + constraint_features + [i / len(waypoints)]
            features.append(feature_vector)
        
        # Pad to fixed length
        while len(features) < self.max_route_length:
            features.append([0] * len(features[0]))
        
        return torch.tensor(features[:self.max_route_length], dtype=torch.float32).unsqueeze(0)
    
    def _normalize_coordinates(self, point: Tuple[float, float]) -> List[float]:
        """Normalize coordinates to [-1, 1] range"""
        lat, lon = point
        # Simple normalization (in production, use proper projection)
        return [lat / 90.0, lon / 180.0]
    
    def _interpolate_waypoints(self, start: List[float], end: List[float], num_points: int) -> List[List[float]]:
        """Interpolate waypoints between start and end"""
        waypoints = []
        for i in range(num_points):
            t = i / (num_points - 1)
            lat = start[0] + t * (end[0] - start[0])
            lon = start[1] + t * (end[1] - start[1])
            waypoints.append([lat, lon])
        return waypoints
    
    def _decode_route(self, 
                     outputs: Dict[str, torch.Tensor],
                     start_point: Tuple[float, float],
                     end_point: Tuple[float, float]) -> Dict:
        """Decode transformer outputs into route format"""
        
        route_coords = outputs['route_coordinates'][0].cpu().numpy()
        traffic_pred = outputs['traffic_prediction'][0].cpu().numpy()
        validity = outputs['route_validity'][0].cpu().numpy()
        
        # Convert normalized coordinates back to lat/lon
        coordinates = []
        for coord in route_coords:
            if validity[0] > 0.5:  # Valid waypoint
                lat = coord[0] * 90.0
                lon = coord[1] * 180.0
                time_estimate = coord[2] * 180.0  # Convert back to minutes
                confidence = coord[3]
                coordinates.append({
                    'lat': float(lat),
                    'lng': float(lon),
                    'time': float(time_estimate),
                    'confidence': float(confidence)
                })
        
        # Calculate route metrics
        total_time = sum(point['time'] for point in coordinates)
        avg_traffic = float(np.mean(traffic_pred))
        
        return {
            'coordinates': coordinates,
            'total_time_minutes': total_time,
            'traffic_level': avg_traffic,
            'quality_score': float(np.mean(validity)),
            'distance_km': self._calculate_distance(coordinates),
            'confidence': float(np.mean([p['confidence'] for p in coordinates]))
        }
    
    def _calculate_distance(self, coordinates: List[Dict]) -> float:
        """Calculate approximate distance in kilometers"""
        if len(coordinates) < 2:
            return 0.0
        
        total_distance = 0.0
        for i in range(1, len(coordinates)):
            # Haversine formula approximation
            lat1, lng1 = coordinates[i-1]['lat'], coordinates[i-1]['lng']
            lat2, lng2 = coordinates[i]['lat'], coordinates[i]['lng']
            
            dlat = np.radians(lat2 - lat1)
            dlng = np.radians(lng2 - lng1)
            
            a = np.sin(dlat/2)**2 + np.cos(np.radians(lat1)) * np.cos(np.radians(lat2)) * np.sin(dlng/2)**2
            c = 2 * np.arcsin(np.sqrt(a))
            
            total_distance += 6371 * c  # Earth radius in km
        
        return total_distance
    
    def save_model(self, path: str):
        """Save the model to disk"""
        model_path = Path(path)
        model_path.mkdir(parents=True, exist_ok=True)
        
        # Save model state
        torch.save(self.state_dict(), model_path / "model.pt")
        
        # Save model configuration
        config = {
            'input_dim': 128,
            'd_model': self.d_model,
            'nhead': 8,
            'num_layers': 6,
            'dim_feedforward': 1024,
            'max_route_length': self.max_route_length
        }
        
        with open(model_path / "config.json", "w") as f:
            json.dump(config, f)
    
    @classmethod
    def load_model(cls, path: str):
        """Load the model from disk"""
        model_path = Path(path)
        
        # Load configuration
        with open(model_path / "config.json", "r") as f:
            config = json.load(f)
        
        # Create model instance
        model = cls(**config)
        
        # Load weights
        model.load_state_dict(torch.load(model_path / "model.pt", map_location='cpu'))
        
        return model


class RouteTransformerTrainer:
    """Trainer class for the Route Transformer model"""
    
    def __init__(self, model: RouteTransformer, learning_rate: float = 1e-4):
        self.model = model
        self.optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            self.optimizer, T_max=1000
        )
        
        # Loss functions
        self.coordinate_loss = nn.MSELoss()
        self.traffic_loss = nn.MSELoss()
        self.preference_loss = nn.CrossEntropyLoss()
        self.validity_loss = nn.BCELoss()
    
    def train_step(self, batch: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Single training step"""
        self.model.train()
        
        # Forward pass
        outputs = self.model(
            batch['route_features'],
            batch.get('attention_mask')
        )
        
        # Calculate losses
        coord_loss = self.coordinate_loss(
            outputs['route_coordinates'], 
            batch['target_coordinates']
        )
        
        traffic_loss = self.traffic_loss(
            outputs['traffic_prediction'], 
            batch['target_traffic']
        )
        
        pref_loss = self.preference_loss(
            outputs['preference_scores'], 
            batch['target_preferences']
        )
        
        validity_loss = self.validity_loss(
            outputs['route_validity'], 
            batch['target_validity']
        )
        
        # Total loss
        total_loss = coord_loss + traffic_loss + pref_loss + validity_loss
        
        # Backward pass
        self.optimizer.zero_grad()
        total_loss.backward()
        torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
        self.optimizer.step()
        self.scheduler.step()
        
        return {
            'total_loss': total_loss.item(),
            'coordinate_loss': coord_loss.item(),
            'traffic_loss': traffic_loss.item(),
            'preference_loss': pref_loss.item(),
            'validity_loss': validity_loss.item()
        }
    
    def validate(self, val_loader) -> Dict[str, float]:
        """Validation step"""
        self.model.eval()
        total_loss = 0.0
        num_batches = 0
        
        with torch.no_grad():
            for batch in val_loader:
                outputs = self.model(
                    batch['route_features'],
                    batch.get('attention_mask')
                )
                
                # Calculate validation loss
                coord_loss = self.coordinate_loss(
                    outputs['route_coordinates'], 
                    batch['target_coordinates']
                )
                
                total_loss += coord_loss.item()
                num_batches += 1
        
        return {'validation_loss': total_loss / num_batches}
