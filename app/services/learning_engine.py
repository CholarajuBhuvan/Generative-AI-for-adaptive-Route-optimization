"""
Adaptive Learning Engine for Continuous Route Optimization Improvement
"""

import asyncio
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from collections import defaultdict, deque
import pickle
from pathlib import Path

from app.core.config import settings


@dataclass
class UserFeedback:
    """User feedback on route quality"""
    route_id: str
    user_id: str
    rating: int  # 1-5 scale
    feedback_type: str  # "time_accuracy", "route_quality", "traffic_accuracy", etc.
    comments: Optional[str] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()


@dataclass
class RoutePerformance:
    """Route performance metrics"""
    route_id: str
    predicted_time: float
    actual_time: float
    predicted_distance: float
    actual_distance: float
    traffic_accuracy: float
    user_satisfaction: float
    timestamp: datetime


@dataclass
class LearningPattern:
    """Learned pattern from user behavior"""
    pattern_id: str
    pattern_type: str  # "time_preference", "route_preference", "traffic_avoidance", etc.
    conditions: Dict[str, Any]  # Conditions when pattern applies
    parameters: Dict[str, float]  # Learned parameters
    confidence: float
    sample_count: int
    last_updated: datetime


class UserPreferenceLearner:
    """Learns individual user preferences from feedback"""
    
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.preferences = {
            'time_weight': 0.4,
            'distance_weight': 0.3,
            'cost_weight': 0.2,
            'scenic_weight': 0.05,
            'traffic_weight': 0.05,
            'avoid_tolls': False,
            'avoid_highways': False,
            'prefer_scenic': False
        }
        self.feedback_history = deque(maxlen=1000)
        self.learning_rate = 0.1
        self.decay_factor = 0.95
    
    def add_feedback(self, feedback: UserFeedback):
        """Add user feedback and update preferences"""
        self.feedback_history.append(feedback)
        self._update_preferences_from_feedback(feedback)
    
    def _update_preferences_from_feedback(self, feedback: UserFeedback):
        """Update preferences based on feedback"""
        # Extract preference signals from feedback
        if feedback.feedback_type == "time_accuracy":
            if feedback.rating >= 4:  # Good time prediction
                # User values time accuracy
                self.preferences['time_weight'] = min(0.8, 
                    self.preferences['time_weight'] + self.learning_rate)
            else:
                # User doesn't care much about time
                self.preferences['time_weight'] = max(0.1, 
                    self.preferences['time_weight'] - self.learning_rate * 0.5)
        
        elif feedback.feedback_type == "route_quality":
            if feedback.rating >= 4:  # Good route
                # User likes the route characteristics
                if "scenic" in feedback.comments.lower():
                    self.preferences['scenic_weight'] = min(0.3, 
                        self.preferences['scenic_weight'] + self.learning_rate)
                if "direct" in feedback.comments.lower():
                    self.preferences['distance_weight'] = min(0.5, 
                        self.preferences['distance_weight'] + self.learning_rate)
        
        elif feedback.feedback_type == "traffic_accuracy":
            if feedback.rating >= 4:  # Good traffic prediction
                self.preferences['traffic_weight'] = min(0.3, 
                    self.preferences['traffic_weight'] + self.learning_rate)
            else:
                self.preferences['traffic_weight'] = max(0.01, 
                    self.preferences['traffic_weight'] - self.learning_rate)
        
        # Normalize preferences to sum to 1.0
        self._normalize_preferences()
    
    def _normalize_preferences(self):
        """Normalize preference weights to sum to 1.0"""
        total_weight = sum([
            self.preferences['time_weight'],
            self.preferences['distance_weight'],
            self.preferences['cost_weight'],
            self.preferences['scenic_weight'],
            self.preferences['traffic_weight']
        ])
        
        if total_weight > 0:
            for key in ['time_weight', 'distance_weight', 'cost_weight', 'scenic_weight', 'traffic_weight']:
                self.preferences[key] /= total_weight
    
    def get_preferences(self) -> Dict[str, float]:
        """Get current user preferences"""
        return self.preferences.copy()
    
    def save_preferences(self, path: str):
        """Save user preferences to disk"""
        data = {
            'user_id': self.user_id,
            'preferences': self.preferences,
            'feedback_count': len(self.feedback_history),
            'last_updated': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
    
    @classmethod
    def load_preferences(cls, user_id: str, path: str) -> 'UserPreferenceLearner':
        """Load user preferences from disk"""
        learner = cls(user_id)
        
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                learner.preferences = data['preferences']
        except FileNotFoundError:
            pass  # Use default preferences
        
        return learner


class TrafficPatternLearner:
    """Learns traffic patterns from historical data"""
    
    def __init__(self):
        self.traffic_patterns = {}
        self.historical_data = defaultdict(list)
        self.pattern_confidence = {}
        self.learning_window = timedelta(days=30)
    
    def add_traffic_data(self, location: Tuple[float, float], 
                        traffic_level: float, timestamp: datetime):
        """Add traffic data point for pattern learning"""
        location_key = f"{location[0]:.4f},{location[1]:.4f}"
        self.historical_data[location_key].append({
            'timestamp': timestamp,
            'traffic_level': traffic_level
        })
        
        # Keep only recent data
        cutoff_time = timestamp - self.learning_window
        self.historical_data[location_key] = [
            data for data in self.historical_data[location_key]
            if data['timestamp'] > cutoff_time
        ]
    
    def learn_patterns(self):
        """Learn traffic patterns from historical data"""
        for location_key, data_points in self.historical_data.items():
            if len(data_points) < 10:  # Need minimum data points
                continue
            
            # Analyze temporal patterns
            patterns = self._analyze_temporal_patterns(data_points)
            
            # Store learned patterns
            self.traffic_patterns[location_key] = patterns
            self.pattern_confidence[location_key] = self._calculate_pattern_confidence(data_points)
    
    def _analyze_temporal_patterns(self, data_points: List[Dict]) -> Dict[str, Any]:
        """Analyze temporal traffic patterns"""
        patterns = {
            'hourly_pattern': [0.0] * 24,
            'daily_pattern': [0.0] * 7,
            'base_traffic': 0.0,
            'variability': 0.0
        }
        
        # Calculate hourly patterns
        hourly_counts = [0] * 24
        hourly_sums = [0.0] * 24
        
        for data in data_points:
            hour = data['timestamp'].hour
            hourly_counts[hour] += 1
            hourly_sums[hour] += data['traffic_level']
        
        for hour in range(24):
            if hourly_counts[hour] > 0:
                patterns['hourly_pattern'][hour] = hourly_sums[hour] / hourly_counts[hour]
        
        # Calculate daily patterns
        daily_counts = [0] * 7
        daily_sums = [0.0] * 7
        
        for data in data_points:
            weekday = data['timestamp'].weekday()
            daily_counts[weekday] += 1
            daily_sums[weekday] += data['traffic_level']
        
        for day in range(7):
            if daily_counts[day] > 0:
                patterns['daily_pattern'][day] = daily_sums[day] / daily_counts[day]
        
        # Calculate base traffic and variability
        traffic_levels = [data['traffic_level'] for data in data_points]
        patterns['base_traffic'] = np.mean(traffic_levels)
        patterns['variability'] = np.std(traffic_levels)
        
        return patterns
    
    def _calculate_pattern_confidence(self, data_points: List[Dict]) -> float:
        """Calculate confidence in learned patterns"""
        if len(data_points) < 20:
            return 0.3
        elif len(data_points) < 50:
            return 0.6
        else:
            return min(0.95, 0.7 + (len(data_points) - 50) / 1000)
    
    def predict_traffic(self, location: Tuple[float, float], 
                       timestamp: datetime) -> Tuple[float, float]:
        """Predict traffic level and confidence"""
        location_key = f"{location[0]:.4f},{location[1]:.4f}"
        
        if location_key not in self.traffic_patterns:
            return 0.5, 0.1  # Default prediction with low confidence
        
        patterns = self.traffic_patterns[location_key]
        confidence = self.pattern_confidence.get(location_key, 0.1)
        
        # Get pattern components
        hour_factor = patterns['hourly_pattern'][timestamp.hour]
        day_factor = patterns['daily_pattern'][timestamp.weekday()]
        base_traffic = patterns['base_traffic']
        
        # Combine factors
        predicted_traffic = (base_traffic + hour_factor + day_factor) / 3.0
        predicted_traffic = max(0.0, min(1.0, predicted_traffic))
        
        return predicted_traffic, confidence
    
    def save_patterns(self, path: str):
        """Save learned patterns to disk"""
        data = {
            'traffic_patterns': self.traffic_patterns,
            'pattern_confidence': self.pattern_confidence,
            'last_updated': datetime.now().isoformat()
        }
        
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
    
    def load_patterns(self, path: str):
        """Load learned patterns from disk"""
        try:
            with open(path, 'r') as f:
                data = json.load(f)
                self.traffic_patterns = data['traffic_patterns']
                self.pattern_confidence = data['pattern_confidence']
        except FileNotFoundError:
            pass  # Start with empty patterns


class RouteAccuracyLearner:
    """Learns route prediction accuracy and improves models"""
    
    def __init__(self):
        self.performance_history = deque(maxlen=10000)
        self.model_accuracy = defaultdict(list)
        self.improvement_suggestions = []
    
    def add_performance_data(self, performance: RoutePerformance):
        """Add route performance data"""
        self.performance_history.append(performance)
        
        # Calculate accuracy metrics
        time_accuracy = 1.0 - abs(performance.predicted_time - performance.actual_time) / max(performance.actual_time, 1.0)
        distance_accuracy = 1.0 - abs(performance.predicted_distance - performance.actual_distance) / max(performance.actual_distance, 0.1)
        
        # Store accuracy by model (would need to track which model was used)
        model_id = "unknown"  # In production, this would come from the route result
        self.model_accuracy[model_id].append({
            'time_accuracy': max(0.0, min(1.0, time_accuracy)),
            'distance_accuracy': max(0.0, min(1.0, distance_accuracy)),
            'traffic_accuracy': performance.traffic_accuracy,
            'user_satisfaction': performance.user_satisfaction,
            'timestamp': performance.timestamp
        })
    
    def analyze_performance(self) -> Dict[str, Any]:
        """Analyze overall performance and generate insights"""
        if not self.performance_history:
            return {'message': 'No performance data available'}
        
        # Calculate overall metrics
        recent_data = [p for p in self.performance_history 
                      if (datetime.now() - p.timestamp).days < 7]
        
        if not recent_data:
            recent_data = list(self.performance_history)
        
        avg_time_accuracy = np.mean([
            1.0 - abs(p.predicted_time - p.actual_time) / max(p.actual_time, 1.0)
            for p in recent_data
        ])
        
        avg_distance_accuracy = np.mean([
            1.0 - abs(p.predicted_distance - p.actual_distance) / max(p.actual_distance, 0.1)
            for p in recent_data
        ])
        
        avg_traffic_accuracy = np.mean([p.traffic_accuracy for p in recent_data])
        avg_user_satisfaction = np.mean([p.user_satisfaction for p in recent_data])
        
        # Generate improvement suggestions
        suggestions = self._generate_improvement_suggestions(
            avg_time_accuracy, avg_distance_accuracy, avg_traffic_accuracy, avg_user_satisfaction
        )
        
        return {
            'overall_metrics': {
                'time_accuracy': avg_time_accuracy,
                'distance_accuracy': avg_distance_accuracy,
                'traffic_accuracy': avg_traffic_accuracy,
                'user_satisfaction': avg_user_satisfaction,
                'total_routes': len(self.performance_history),
                'recent_routes': len(recent_data)
            },
            'improvement_suggestions': suggestions,
            'model_performance': dict(self.model_accuracy)
        }
    
    def _generate_improvement_suggestions(self, time_acc: float, distance_acc: float, 
                                        traffic_acc: float, user_sat: float) -> List[str]:
        """Generate improvement suggestions based on performance"""
        suggestions = []
        
        if time_acc < 0.7:
            suggestions.append("Time prediction accuracy is low - consider improving traffic data integration")
        
        if distance_acc < 0.8:
            suggestions.append("Distance prediction needs improvement - check routing algorithms")
        
        if traffic_acc < 0.6:
            suggestions.append("Traffic prediction is inaccurate - update traffic models")
        
        if user_sat < 0.7:
            suggestions.append("User satisfaction is low - review route preferences and alternatives")
        
        if time_acc < 0.6 and traffic_acc < 0.6:
            suggestions.append("Both time and traffic predictions are poor - consider model retraining")
        
        if not suggestions:
            suggestions.append("Performance metrics look good - continue monitoring")
        
        return suggestions


class AdaptiveLearningEngine:
    """
    Main adaptive learning engine that coordinates all learning components
    """
    
    def __init__(self):
        self.user_learners: Dict[str, UserPreferenceLearner] = {}
        self.traffic_learner = TrafficPatternLearner()
        self.route_learner = RouteAccuracyLearner()
        
        self.learning_enabled = True
        self.learning_interval = timedelta(hours=1)
        self.last_learning_update = datetime.now()
        
        # Learning storage paths
        self.data_dir = Path("learning_data")
        self.data_dir.mkdir(exist_ok=True)
    
    async def initialize(self):
        """Initialize the learning engine"""
        try:
            # Load existing learned patterns
            self.traffic_learner.load_patterns(self.data_dir / "traffic_patterns.json")
            
            # Load user preferences for existing users
            for user_file in self.data_dir.glob("user_*.json"):
                user_id = user_file.stem.replace("user_", "")
                self.user_learners[user_id] = UserPreferenceLearner.load_preferences(
                    user_id, str(user_file)
                )
            
            logging.info(f"Adaptive Learning Engine initialized with {len(self.user_learners)} users")
            
        except Exception as e:
            logging.error(f"Error initializing learning engine: {e}")
    
    async def process_user_feedback(self, feedback: UserFeedback):
        """Process user feedback for learning"""
        if not self.learning_enabled:
            return
        
        # Get or create user learner
        if feedback.user_id not in self.user_learners:
            self.user_learners[feedback.user_id] = UserPreferenceLearner(feedback.user_id)
        
        # Add feedback to user learner
        self.user_learners[feedback.user_id].add_feedback(feedback)
        
        # Save updated preferences
        self.user_learners[feedback.user_id].save_preferences(
            self.data_dir / f"user_{feedback.user_id}.json"
        )
        
        logging.info(f"Processed feedback for user {feedback.user_id}")
    
    async def process_traffic_data(self, location: Tuple[float, float], 
                                 traffic_level: float, timestamp: datetime):
        """Process traffic data for pattern learning"""
        if not self.learning_enabled:
            return
        
        self.traffic_learner.add_traffic_data(location, traffic_level, timestamp)
    
    async def process_route_performance(self, performance: RoutePerformance):
        """Process route performance data"""
        if not self.learning_enabled:
            return
        
        self.route_learner.add_performance_data(performance)
    
    async def get_user_preferences(self, user_id: str) -> Dict[str, float]:
        """Get learned preferences for a user"""
        if user_id in self.user_learners:
            return self.user_learners[user_id].get_preferences()
        else:
            # Return default preferences for new users
            return {
                'time_weight': 0.4,
                'distance_weight': 0.3,
                'cost_weight': 0.2,
                'scenic_weight': 0.05,
                'traffic_weight': 0.05
            }
    
    async def predict_traffic(self, location: Tuple[float, float], 
                            timestamp: datetime) -> Tuple[float, float]:
        """Predict traffic using learned patterns"""
        return self.traffic_learner.predict_traffic(location, timestamp)
    
    async def run_learning_cycle(self):
        """Run a complete learning cycle"""
        if not self.learning_enabled:
            return
        
        try:
            logging.info("Running adaptive learning cycle...")
            
            # Learn traffic patterns
            self.traffic_learner.learn_patterns()
            self.traffic_learner.save_patterns(self.data_dir / "traffic_patterns.json")
            
            # Update learning timestamp
            self.last_learning_update = datetime.now()
            
            logging.info("Learning cycle completed successfully")
            
        except Exception as e:
            logging.error(f"Error in learning cycle: {e}")
    
    async def get_learning_analytics(self) -> Dict[str, Any]:
        """Get comprehensive learning analytics"""
        performance_analysis = self.route_learner.analyze_performance()
        
        return {
            'learning_status': {
                'enabled': self.learning_enabled,
                'last_update': self.last_learning_update.isoformat(),
                'users_tracked': len(self.user_learners),
                'traffic_patterns': len(self.traffic_learner.traffic_patterns)
            },
            'performance_analysis': performance_analysis,
            'traffic_patterns': {
                'learned_locations': len(self.traffic_learner.traffic_patterns),
                'pattern_confidence': {
                    location: confidence 
                    for location, confidence in self.traffic_learner.pattern_confidence.items()
                }
            },
            'user_insights': {
                user_id: {
                    'feedback_count': len(learner.feedback_history),
                    'preferences': learner.get_preferences()
                }
                for user_id, learner in self.user_learners.items()
            }
        }
    
    async def enable_learning(self):
        """Enable adaptive learning"""
        self.learning_enabled = True
        logging.info("Adaptive learning enabled")
    
    async def disable_learning(self):
        """Disable adaptive learning"""
        self.learning_enabled = False
        logging.info("Adaptive learning disabled")
    
    async def reset_learning_data(self, user_id: str = None):
        """Reset learning data for a user or all users"""
        if user_id:
            if user_id in self.user_learners:
                del self.user_learners[user_id]
                user_file = self.data_dir / f"user_{user_id}.json"
                if user_file.exists():
                    user_file.unlink()
                logging.info(f"Reset learning data for user {user_id}")
        else:
            # Reset all learning data
            self.user_learners.clear()
            self.traffic_learner = TrafficPatternLearner()
            self.route_learner = RouteAccuracyLearner()
            
            # Remove all learning files
            for file in self.data_dir.glob("*.json"):
                file.unlink()
            
            logging.info("Reset all learning data")
    
    async def schedule_learning_tasks(self):
        """Schedule periodic learning tasks"""
        while True:
            try:
                await asyncio.sleep(self.learning_interval.total_seconds())
                await self.run_learning_cycle()
            except Exception as e:
                logging.error(f"Error in scheduled learning task: {e}")
                await asyncio.sleep(300)  # Wait 5 minutes before retrying


# Global learning engine instance
learning_engine = AdaptiveLearningEngine()
