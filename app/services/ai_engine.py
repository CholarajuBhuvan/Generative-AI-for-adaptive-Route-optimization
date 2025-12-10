"""
Main AI Engine that orchestrates all AI models for route optimization
"""

import asyncio
import numpy as np
import requests
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime, timedelta
import logging
from dataclasses import dataclass
import json
from pathlib import Path

from app.services.traffic_service import traffic_service
from app.core.config import settings
from app.models.route_models import RouteRequest, RouteResult


class AIEngine:
    """
    Main AI Engine that coordinates all AI models for route optimization
    """
    
    def __init__(self):
        # Lazy-imported models to avoid heavy deps blocking startup
        self.transformer_model: Optional[Any] = None
        self.rl_agent: Optional[Any] = None
        self.genetic_optimizer: Optional[Any] = None
        
        # Import FastAIEngine with road-based routing
        from app.services.fast_ai_engine import FastAIEngine
        self.fast_engine = FastAIEngine()
        
        self.model_cache = {}
        self.request_history = []
        self.performance_metrics = {}
        
        self.is_initialized = False
        self.use_fast_mode = True  # Default to fast mode for better UX
    
    async def initialize(self):
        """Initialize all AI models"""
        try:
            logging.info("Initializing AI Engine...")
            
            # Always initialize fast engine first for quick responses
            await self.fast_engine.initialize()
            
            # Initialize other models in the background (non-blocking)
            try:
                # Initialize Transformer Model
                await self._initialize_transformer_model()
                
                # Initialize RL Agent  
                await self._initialize_rl_agent()
                
                # Initialize Genetic Algorithm
                await self._initialize_genetic_optimizer()
                
                logging.info("All AI models initialized successfully")
            except Exception as model_error:
                logging.warning(f"Some AI models failed to initialize: {model_error}")
                logging.info("Falling back to fast engine only")
            
            self.is_initialized = True
            logging.info("AI Engine initialized successfully")
            
        except Exception as e:
            logging.error(f"Failed to initialize AI Engine: {e}")
            raise
    
    async def _initialize_transformer_model(self):
        """Initialize the transformer-based route generation model"""
        try:
            # Try PyTorch-based transformer first, fallback to simple version
            try:
                from app.models.transformer_model import RouteTransformer
                model_path = settings.transformer_model_path
                if Path(model_path).exists():
                    self.transformer_model = RouteTransformer.load_model(model_path)
                    logging.info("Loaded pre-trained transformer model")
                else:
                    self.transformer_model = RouteTransformer(
                        input_dim=128, d_model=256, nhead=8, num_layers=6, max_route_length=100
                    )
                    logging.info("Created new transformer model")
            except Exception as torch_error:
                logging.warning(f"PyTorch transformer failed: {torch_error}, using simple version")
                # Fallback to simple transformer without PyTorch dependency
                from app.models.simple_transformer import SimpleTransformerModel
                self.transformer_model = SimpleTransformerModel(
                    input_dim=128, d_model=256, num_heads=8
                )
                self.transformer_model.initialize()
                logging.info("Initialized simple transformer model")
                
        except Exception as e:
            logging.error(f"Error initializing any transformer model: {e}")
            self.transformer_model = None
    
    async def _initialize_rl_agent(self):
        """Initialize the reinforcement learning agent"""
        try:
            # Try PyTorch-based RL agent first, fallback to simple version
            try:
                from app.models.rl_agent import RLAgent
                model_path = settings.rl_model_path
                if Path(model_path).exists():
                    self.rl_agent = RLAgent.load_model(model_path)
                    logging.info("Loaded pre-trained RL agent")
                else:
                    self.rl_agent = RLAgent(
                        state_dim=32, action_dim=8, learning_rate=1e-4, gamma=0.99
                    )
                    logging.info("Created new RL agent")
            except Exception as torch_error:
                logging.warning(f"PyTorch RL agent failed: {torch_error}, using simple version")
                # Fallback to simple RL agent without PyTorch dependency
                from app.models.simple_rl_agent import SimpleRLAgent
                self.rl_agent = SimpleRLAgent(
                    state_dim=32, action_dim=8, learning_rate=0.1, gamma=0.9
                )
                self.rl_agent.initialize()
                logging.info("Initialized simple RL agent")
                
        except Exception as e:
            logging.error(f"Error initializing any RL agent: {e}")
            self.rl_agent = None
    
    async def _initialize_genetic_optimizer(self):
        """Initialize the genetic algorithm optimizer"""
        try:
            from app.models.genetic_algorithm import GeneticRouteOptimizer, OptimizationObjectives
            objectives = OptimizationObjectives(
                minimize_time=True,
                minimize_distance=True,
                minimize_cost=True,
                maximize_scenic_value=False,
                minimize_traffic=True,
                time_weight=0.4,
                distance_weight=0.3,
                cost_weight=0.2,
                scenic_weight=0.05,
                traffic_weight=0.05
            )
            
            self.genetic_optimizer = GeneticRouteOptimizer(
                population_size=settings.genetic_population_size,
                max_generations=settings.genetic_generations,
                mutation_rate=settings.genetic_mutation_rate,
                objectives=objectives
            )
            logging.info("Created genetic algorithm optimizer")
        except Exception as e:
            logging.error(f"Error initializing genetic optimizer: {e}")
            self.genetic_optimizer = None
    
    async def optimize_route(self, request: RouteRequest) -> RouteResult:
        """
        Optimize route using the best AI model for the given request
        
        Args:
            request: Route optimization request
            
        Returns:
            Optimized route result
        """
        if not self.is_initialized:
            raise RuntimeError("AI Engine not initialized")
        
        # Use fast mode only if no AI models are available
        if not self._heavy_models_available():
            logging.info("No AI models available, using fast AI engine for route optimization")
            return await self.fast_engine.optimize_route(request)
        
        # Fallback to complex AI models only if specifically requested
        try:
            # Get traffic data
            traffic_data = await self._get_traffic_data_for_route(request)
            
            # Select best AI model for this request
            selected_model = self._select_best_model(request, traffic_data)
            
            # Generate route using selected model
            route_data = await self._generate_route_with_model(
                selected_model, request, traffic_data
            )
        except Exception as e:
            logging.warning(f"Complex AI models failed: {e}, falling back to fast engine")
            return await self.fast_engine.optimize_route(request)

        # Snap the primary route to roads using OSRM for map-accurate geometry
        snapped_main = self._snap_route_to_road(route_data['coordinates'])
        if snapped_main and len(snapped_main) > 1:
            route_data['coordinates'] = snapped_main
            # Recalculate distance based on snapped coordinates
            total_distance_km = 0.0
            for i in range(len(snapped_main) - 1):
                p1 = (snapped_main[i]['lat'], snapped_main[i]['lng'])
                p2 = (snapped_main[i+1]['lat'], snapped_main[i+1]['lng'])
                total_distance_km += self._calculate_distance(p1, p2)
            route_data['total_distance_km'] = total_distance_km
            # Use OSRM's accurate time from the last waypoint (already calculated in snap function)
            route_data['total_time_minutes'] = snapped_main[-1]['time']
        else:
            # If snapping failed, recalculate time based on actual distance
            if 'total_distance_km' in route_data:
                route_data['total_time_minutes'] = self._calculate_realistic_time(
                    route_data['total_distance_km'], request.travel_mode, traffic_data
                )
        
        # Analyze traffic conditions
        traffic_analysis = await self._analyze_route_traffic(route_data['coordinates'])
        
        # Generate alternative routes with timeout
        try:
            alternatives = await asyncio.wait_for(
                self._generate_alternative_routes(request, traffic_data),
                timeout=30.0  # Max 30 seconds for all 3 alternative models
            )
        except (asyncio.TimeoutError, Exception) as e:
            logging.warning(f"Alternative routes timeout: {e}")
            alternatives = []  # Skip alternatives if too slow
        
        # Create result
        result = RouteResult(
            route_id=f"route_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{np.random.randint(1000, 9999)}",
            coordinates=route_data['coordinates'],
            total_distance_km=route_data['total_distance_km'],
            total_time_minutes=route_data['total_time_minutes'],
            total_cost=route_data.get('total_cost', 0.0),
            confidence_score=route_data['confidence_score'],
            ai_model_used=selected_model,
            traffic_analysis=traffic_analysis,
            alternatives=alternatives,
            metadata={
                'generation_time': datetime.now().isoformat(),
                'request_constraints': request.constraints,
                'user_preferences': request.user_preferences,
                'travel_mode': request.travel_mode
            }
        )
        
        # Store in history for learning
        self._store_route_result(result)
        
        return result
    
    async def _get_traffic_data_for_route(self, request: RouteRequest) -> Dict[str, Any]:
        """Get comprehensive traffic data for route optimization with timeouts"""
        # Get traffic data with short timeouts to prevent hanging
        try:
            start_traffic = await asyncio.wait_for(
                traffic_service.get_comprehensive_traffic_data(request.start_point, radius=5.0),
                timeout=1.0
            )
        except (asyncio.TimeoutError, Exception) as e:
            logging.warning(f"Start traffic timeout: {e}")
            start_traffic = {'traffic_level': 0.5}
        
        try:
            end_traffic = await asyncio.wait_for(
                traffic_service.get_comprehensive_traffic_data(request.end_point, radius=5.0),
                timeout=1.0
            )
        except (asyncio.TimeoutError, Exception) as e:
            logging.warning(f"End traffic timeout: {e}")
            end_traffic = {'traffic_level': 0.5}
        
        # Get traffic data for midpoint
        mid_lat = (request.start_point[0] + request.end_point[0]) / 2
        mid_lng = (request.start_point[1] + request.end_point[1]) / 2
        try:
            mid_traffic = await asyncio.wait_for(
                traffic_service.get_comprehensive_traffic_data((mid_lat, mid_lng), radius=10.0),
                timeout=1.0
            )
        except (asyncio.TimeoutError, Exception) as e:
            logging.warning(f"Mid traffic timeout: {e}")
            mid_traffic = {'traffic_level': 0.5}
        
        # Safely extract traffic levels with fallbacks
        start_level = self._extract_traffic_level(start_traffic)
        end_level = self._extract_traffic_level(end_traffic)
        mid_level = self._extract_traffic_level(mid_traffic)
        
        return {
            'start_traffic': start_traffic,
            'end_traffic': end_traffic,
            'mid_traffic': mid_traffic,
            'overall_traffic_level': np.mean([start_level, end_level, mid_level])
        }
    
    def _extract_traffic_level(self, traffic_data: Dict[str, Any]) -> float:
        """Safely extract traffic level from traffic data with fallbacks"""
        try:
            # Try multiple possible structures
            if 'traffic_metrics' in traffic_data and 'level' in traffic_data['traffic_metrics']:
                return float(traffic_data['traffic_metrics']['level'])
            elif 'traffic_level' in traffic_data:
                return float(traffic_data['traffic_level'])
            elif 'level' in traffic_data:
                return float(traffic_data['level'])
            else:
                # Default moderate traffic level
                return 0.5
        except (KeyError, TypeError, ValueError):
            # Fallback to moderate traffic level
            return 0.5
    
    def _select_best_model(self, request: RouteRequest, traffic_data: Dict[str, Any]) -> str:
        """
        Select the best AI model based on request characteristics and traffic conditions
        
        Args:
            request: Route optimization request
            traffic_data: Current traffic conditions
            
        Returns:
            Name of the best model to use
        """
        # Decision logic based on request characteristics
        distance = self._calculate_distance(request.start_point, request.end_point)
        traffic_level = traffic_data['overall_traffic_level']
        
        # Choose available model based on conditions and availability
        candidates: List[str] = []
        if self.transformer_model is not None:
            candidates.append("transformer")
        if self.rl_agent is not None:
            candidates.append("rl_agent")
        if self.genetic_optimizer is not None:
            candidates.append("genetic")

        # Fallback to genetic if others unavailable
        if not candidates and self.genetic_optimizer is None:
            # As a last resort, prefer transformer if available; else rl; else genetic
            return "genetic"

        # Enhanced preference logic to better utilize all three models
        
        # High traffic scenarios - use RL agent for adaptive decision making
        if traffic_level > 0.6 and "rl_agent" in candidates:
            logging.info(f"Selected RL Agent due to high traffic level: {traffic_level}")
            return "rl_agent"
        
        # Complex constraints - use genetic algorithm for multi-objective optimization  
        if len(request.constraints) > 2 and "genetic" in candidates:
            logging.info(f"Selected Genetic Algorithm due to complex constraints: {len(request.constraints)}")
            return "genetic"
        
        # Long distance routes - use genetic algorithm for global optimization
        if distance > 25.0 and "genetic" in candidates:
            logging.info(f"Selected Genetic Algorithm due to long distance: {distance}km")
            return "genetic"
        
        # User preferences complexity - use genetic for multi-objective optimization
        if request.user_preferences and len(request.user_preferences) > 2 and "genetic" in candidates:
            logging.info("Selected Genetic Algorithm due to complex user preferences")
            return "genetic"
        
        # Medium traffic with time-sensitive preferences - use RL agent
        if (traffic_level > 0.4 and request.user_preferences and 
            request.user_preferences.get('time_weight', 0) > 0.6 and "rl_agent" in candidates):
            logging.info("Selected RL Agent for time-sensitive route in moderate traffic")
            return "rl_agent"
        
        # Short to medium distance with moderate complexity - use transformer
        if distance <= 25.0 and "transformer" in candidates:
            logging.info(f"Selected Transformer for moderate distance route: {distance}km")
            return "transformer"
        
        # Fallback: rotate between available models for variety
        import time
        model_rotation = int(time.time()) % len(candidates) if candidates else 0
        selected = candidates[model_rotation] if candidates else "genetic"
        logging.info(f"Selected {selected} via rotation fallback")
        return selected
    
    async def _generate_route_with_model(self, 
                                       model_name: str,
                                       request: RouteRequest,
                                       traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate route using the specified AI model"""
        
        if model_name == "transformer" and self.transformer_model is not None:
            return await self._generate_route_with_transformer(request, traffic_data)
        elif model_name == "rl_agent" and self.rl_agent is not None:
            return await self._generate_route_with_rl(request, traffic_data)
        elif model_name == "genetic" and self.genetic_optimizer is not None:
            return await self._generate_route_with_genetic(request, traffic_data)
        else:
            raise ValueError(f"Unknown model: {model_name}")
    
    async def _generate_route_with_transformer(self, 
                                             request: RouteRequest,
                                             traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate route using transformer model"""
        try:
            logging.info("Generating route with Transformer AI model")
            
            # Check if it's our simple transformer model
            if hasattr(self.transformer_model, 'optimize_route'):
                # Use simple transformer model
                result = self.transformer_model.optimize_route(
                    request.start_point,
                    request.end_point,
                    traffic_data,
                    request.user_preferences or {}
                )
                
                return {
                    'coordinates': result['coordinates'],
                    'total_distance_km': result['total_distance_km'],
                    'total_time_minutes': result['total_time_minutes'],
                    'total_cost': result['total_distance_km'] * 8.5,  # Basic cost calculation
                    'confidence_score': result['confidence_score']
                }
            else:
                # Use PyTorch transformer model (original logic)
                constraints = {
                    'max_time': request.constraints.get('max_time', 60),
                    'max_distance': request.constraints.get('max_distance', 50),
                    'avoid_tolls': request.constraints.get('avoid_tolls', False),
                    'avoid_highways': request.constraints.get('avoid_highways', False),
                    'prefer_scenic': request.constraints.get('prefer_scenic', False)
                }
                
                routes = self.transformer_model.generate_route(
                    request.start_point,
                    request.end_point,
                    constraints,
                    num_alternatives=1
                )
                
                if routes:
                    best_route = routes[0]
                    return {
                        'coordinates': best_route['coordinates'],
                        'total_distance_km': best_route['distance_km'],
                        'total_time_minutes': best_route['total_time_minutes'],
                        'total_cost': 0.0,
                        'confidence_score': best_route['confidence']
                    }
                else:
                    raise ValueError("Transformer model failed to generate route")
                
        except Exception as e:
            logging.error(f"Error generating route with transformer: {e}")
            return self._generate_fallback_route(request)
    
    async def _generate_route_with_rl(self, 
                                    request: RouteRequest,
                                    traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate route using RL agent"""
        try:
            logging.info("Generating route with RL Agent AI model")
            
            # Check if it's our simple RL agent model
            if hasattr(self.rl_agent, 'optimize_route'):
                # Use simple RL agent model
                result = self.rl_agent.optimize_route(
                    request.start_point,
                    request.end_point,
                    traffic_data,
                    request.user_preferences or {}
                )
                
                return {
                    'coordinates': result['coordinates'],
                    'total_distance_km': result['total_distance_km'],
                    'total_time_minutes': result['total_time_minutes'],
                    'total_cost': result['total_distance_km'] * 8.5,  # Basic cost calculation
                    'confidence_score': result['confidence_score']
                }
            else:
                # Use PyTorch RL agent (original logic)
                map_data = {'roads': [], 'obstacles': []}
                
                result = self.rl_agent.optimize_route(
                    request.start_point,
                    request.end_point,
                    map_data,
                    traffic_data,
                    num_episodes=50
                )
                
                coordinates = []
                total_distance = 0.0
                total_time = 0.0
                
                for i, point in enumerate(result['route']):
                    coordinates.append({
                        'lat': point['lat'],
                        'lng': point['lng'],
                        'time': point.get('time', i * 2),
                        'confidence': 0.8
                    })
                    
                    if i > 0:
                        prev_point = result['route'][i-1]
                        distance = self._calculate_distance(
                            (prev_point['lat'], prev_point['lng']),
                            (point['lat'], point['lng'])
                        )
                        total_distance += distance
                
                total_time = len(result['route']) * 2
                
                return {
                    'coordinates': coordinates,
                    'total_distance_km': total_distance,
                    'total_time_minutes': total_time,
                    'total_cost': total_distance * 0.5,
                    'confidence_score': 0.8
                }
            
        except Exception as e:
            logging.error(f"Error generating route with RL agent: {e}")
            return self._generate_fallback_route(request)
    
    async def _generate_route_with_genetic(self, 
                                         request: RouteRequest,
                                         traffic_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate route using genetic algorithm"""
        try:
            # Lazy import to avoid heavy deps at module import time
            from app.models.genetic_algorithm import GeneticRouteOptimizer, OptimizationObjectives
            # Prepare optimization objectives
            objectives = OptimizationObjectives(
                minimize_time=request.user_preferences.get('time_weight', 0.4) > 0,
                minimize_distance=request.user_preferences.get('distance_weight', 0.3) > 0,
                minimize_cost=request.user_preferences.get('cost_weight', 0.2) > 0,
                maximize_scenic_value=request.user_preferences.get('scenic_weight', 0.1) > 0,
                minimize_traffic=request.user_preferences.get('traffic_weight', 0.1) > 0,
                time_weight=request.user_preferences.get('time_weight', 0.4),
                distance_weight=request.user_preferences.get('distance_weight', 0.3),
                cost_weight=request.user_preferences.get('cost_weight', 0.2),
                scenic_weight=request.user_preferences.get('scenic_weight', 0.1),
                traffic_weight=request.user_preferences.get('traffic_weight', 0.1)
            )
            
            # Create optimizer with custom objectives
            optimizer = GeneticRouteOptimizer(
                population_size=50,  # Reduced for faster response
                max_generations=30,
                mutation_rate=0.1,
                objectives=objectives
            )
            
            # Optimize route
            result = optimizer.optimize_route(
                request.start_point,
                request.end_point,
                request.constraints,
                traffic_data,
                {'roads': [], 'obstacles': []}  # Simplified map data
            )
            
            return {
                'coordinates': result['route']['coordinates'],
                'total_distance_km': result['route']['total_distance_km'],
                'total_time_minutes': result['route']['total_time_minutes'],
                'total_cost': result['route']['total_cost'],
                'confidence_score': result['route']['fitness_score']
            }
            
        except Exception as e:
            logging.error(f"Error generating route with genetic algorithm: {e}")
            return self._generate_fallback_route(request)
    
    def _generate_fallback_route(self, request: RouteRequest) -> Dict[str, Any]:
        """Generate a simple fallback route when AI models fail"""
        # Create a simple straight-line route with interpolated waypoints
        num_waypoints = 12
        coordinates = []
        
        for i in range(num_waypoints):
            t = i / (num_waypoints - 1)
            # Add tiny jitter for visual variety
            jitter_lat = (np.random.rand() - 0.5) * 0.002
            jitter_lng = (np.random.rand() - 0.5) * 0.002
            lat = request.start_point[0] + t * (request.end_point[0] - request.start_point[0]) + jitter_lat
            lng = request.start_point[1] + t * (request.end_point[1] - request.start_point[1]) + jitter_lng
            
            coordinates.append({
                'lat': lat,
                'lng': lng,
                'time': i * 5,  # 5 minutes per waypoint
                'confidence': 0.5
            })
        
        # Try to snap to roads using OSRM
        snapped = self._snap_route_to_road(coordinates)

        total_distance = self._calculate_distance(request.start_point, request.end_point)
        
        return {
            'coordinates': snapped if snapped else coordinates,
            'total_distance_km': total_distance,
            'total_time_minutes': num_waypoints * 4,
            'total_cost': total_distance * 0.5,
            'confidence_score': 0.5
        }
    
    async def _analyze_route_traffic(self, coordinates: List[Dict[str, float]]) -> Dict[str, Any]:
        """Analyze traffic conditions along the route"""
        try:
            # Convert coordinates to tuple format
            route_points = [(coord['lat'], coord['lng']) for coord in coordinates]
            
            # Get traffic analysis with timeout
            analysis = await asyncio.wait_for(
                traffic_service.get_route_traffic_analysis(route_points),
                timeout=2.0
            )
            
            return analysis
        except Exception as e:
            logging.error(f"Error analyzing route traffic: {e}")
            return {
                'error': str(e),
                'segments': [],
                'overall_metrics': {},
                'bottlenecks': [],
                'recommendations': ['Unable to analyze traffic conditions']
            }
    
    async def _generate_alternative_routes(self, 
                                         request: RouteRequest,
                                         traffic_data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate alternative routes using different models"""
        alternatives: List[Dict[str, Any]] = []

        # Preferred: try different models if available
        for model_name in ["transformer", "rl_agent", "genetic"]:
            try:
                route_data = await self._generate_route_with_model(model_name, request, traffic_data)
                # Snap generated route to roads if possible
                snapped = self._snap_route_to_road(route_data['coordinates'])
                if snapped and len(snapped) > 1:
                    route_data['coordinates'] = snapped
                    # Recalculate distance for alternative
                    total_distance_km = 0.0
                    for i in range(len(snapped) - 1):
                        p1 = (snapped[i]['lat'], snapped[i]['lng'])
                        p2 = (snapped[i+1]['lat'], snapped[i+1]['lng'])
                        total_distance_km += self._calculate_distance(p1, p2)
                    route_data['total_distance_km'] = total_distance_km
                    # Use OSRM's accurate time from snapped waypoints
                    route_data['total_time_minutes'] = snapped[-1]['time']
                alternatives.append({
                    'model_used': model_name,
                    'coordinates': route_data['coordinates'],
                    'total_distance_km': route_data['total_distance_km'],
                    'total_time_minutes': route_data['total_time_minutes'],
                    'confidence_score': route_data['confidence_score']
                })
            except Exception:
                continue
            if len(alternatives) >= 3:
                return alternatives

        # Fallback: generate diverse variations using fallback/genetic
        try:
            # Attempt genetic three times with slight parameter variations
            for idx in range(3 - len(alternatives)):
                varied_request = RouteRequest(
                    start_point=request.start_point,
                    end_point=request.end_point,
                    constraints={**request.constraints, 'variation': idx},
                    user_preferences={**request.user_preferences}
                )
                route_data = await self._generate_route_with_genetic(varied_request, traffic_data)
                snapped = self._snap_route_to_road(route_data['coordinates'])
                if snapped and len(snapped) > 1:
                    route_data['coordinates'] = snapped
                    total_distance_km = 0.0
                    for i in range(len(snapped) - 1):
                        p1 = (snapped[i]['lat'], snapped[i]['lng'])
                        p2 = (snapped[i+1]['lat'], snapped[i+1]['lng'])
                        total_distance_km += self._calculate_distance(p1, p2)
                    route_data['total_distance_km'] = total_distance_km
                    # Use OSRM's accurate time from snapped waypoints
                    route_data['total_time_minutes'] = snapped[-1]['time']
                alternatives.append({
                    'model_used': f'genetic_var_{idx+1}',
                    'coordinates': route_data['coordinates'],
                    'total_distance_km': route_data['total_distance_km'],
                    'total_time_minutes': route_data['total_time_minutes'],
                    'confidence_score': route_data['confidence_score']
                })
                if len(alternatives) >= 3:
                    return alternatives
        except Exception:
            pass

        # Last resort: multiple fallback routes with different jitter
        for idx in range(3 - len(alternatives)):
            route_data = self._generate_fallback_route(request)
            alternatives.append({
                'model_used': f'fallback_{idx+1}',
                'coordinates': route_data['coordinates'],
                'total_distance_km': route_data['total_distance_km'],
                'total_time_minutes': route_data['total_time_minutes'],
                'confidence_score': route_data['confidence_score']
            })

        return alternatives

    def _snap_route_to_road(self, coordinates: List[Dict[str, float]]) -> List[Dict[str, float]]:
        """Snap a sequence of coordinates to roads using OSRM public API.

        Returns a list of coordinates if successful; otherwise returns empty list.
        """
        try:
            if not coordinates or len(coordinates) < 2:
                return []
            # OSRM expects lon,lat;lon,lat
            coords_param = ";".join([
                f"{c['lng']},{c['lat']}" for c in coordinates
            ])
            url = (
                f"https://router.project-osrm.org/route/v1/driving/" 
                f"{coords_param}?overview=full&geometries=geojson&steps=false"
            )
            resp = requests.get(url, timeout=6)
            if resp.status_code != 200:
                return []
            data = resp.json()
            if not data.get('routes'):
                return []
            
            route = data['routes'][0]
            geometry = route['geometry']
            # OSRM provides accurate duration in seconds - use this!
            total_duration_minutes = route.get('duration', 0) / 60.0  # Convert seconds to minutes
            
            # geometry['coordinates'] is a list of [lon, lat]
            snapped_coords = [
                { 'lat': lonlat[1], 'lng': lonlat[0], 'time': 0.0, 'confidence': 0.95 }
                for lonlat in geometry['coordinates']
            ]
            
            # Distribute time proportionally across waypoints based on OSRM duration
            if len(snapped_coords) > 1:
                for idx in range(len(snapped_coords)):
                    # Proportional time based on position in route
                    snapped_coords[idx]['time'] = (idx / (len(snapped_coords) - 1)) * total_duration_minutes
            
            return snapped_coords
        except Exception as e:
            logging.warning(f"Road snapping failed: {e}")
            return []
    
    def _calculate_distance(self, point1: Tuple[float, float], point2: Tuple[float, float]) -> float:
        """Calculate distance between two points using Haversine formula"""
        lat1, lon1 = point1
        lat2, lon2 = point2
        
        # Convert latitude and longitude to radians
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        delta_lat = np.radians(lat2 - lat1)
        delta_lon = np.radians(lon2 - lon1)
        
        # Haversine formula
        a = np.sin(delta_lat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(delta_lon/2)**2
        c = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
        
        # Earth's radius in kilometers
        R = 6371
        distance = R * c
        
        return distance
    
    def _calculate_realistic_time(self, distance_km: float, travel_mode: str, 
                                traffic_data: Dict[str, Any] = None) -> float:
        """
        Calculate realistic travel time based on mode, distance, and traffic
        
        Args:
            distance_km: Distance in kilometers
            travel_mode: Mode of travel
            traffic_data: Current traffic conditions
            
        Returns:
            Time in minutes
        """
        # Base speeds (km/h) for different travel modes
        base_speeds = {
            'driving': 50,      # City driving
            'cycling': 15,      # Cycling speed
            'walking': 5,       # Walking speed
            'transit': 25       # Public transport average
        }
        
        base_speed = base_speeds.get(travel_mode, 50)
        
        # Adjust for traffic if available
        if traffic_data and 'overall_traffic_level' in traffic_data:
            traffic_factor = traffic_data['overall_traffic_level']
            # Heavy traffic reduces speed
            speed_reduction = traffic_factor * 0.4  # Max 40% reduction
            adjusted_speed = base_speed * (1 - speed_reduction)
        else:
            # Default traffic adjustment for different times
            current_hour = datetime.now().hour
            if 7 <= current_hour <= 9 or 17 <= current_hour <= 19:  # Rush hours
                adjusted_speed = base_speed * 0.7  # 30% slower
            elif 10 <= current_hour <= 16:  # Daytime
                adjusted_speed = base_speed * 0.85  # 15% slower
            else:  # Off-peak
                adjusted_speed = base_speed * 0.95  # 5% slower
        
        # Calculate base time
        base_time_hours = distance_km / max(adjusted_speed, 5)  # Min 5 km/h
        
        # Add delays based on distance (longer routes have more delays)
        if distance_km > 50:
            delay_factor = 1 + (distance_km - 50) * 0.005  # 0.5% per km over 50km
        else:
            delay_factor = 1.1  # 10% base delay for stops, lights, etc.
        
        # Convert to minutes
        total_time_minutes = base_time_hours * 60 * delay_factor
        
        # Minimum time based on distance (can't be too fast)
        min_time = distance_km * 0.8  # At least 0.8 minutes per km
        
        return max(total_time_minutes, min_time, 5.0)
    
    def _store_route_result(self, result: RouteResult):
        """Store route result for learning and analytics"""
        self.request_history.append({
            'timestamp': datetime.now(),
            'route_id': result.route_id,
            'model_used': result.ai_model_used,
            'performance': {
                'confidence': result.confidence_score,
                'distance': result.total_distance_km,
                'time': result.total_time_minutes
            }
        })
        
        # Keep only recent history
        if len(self.request_history) > 1000:
            self.request_history = self.request_history[-1000:]
    
    async def get_performance_analytics(self) -> Dict[str, Any]:
        """Get performance analytics for AI models"""
        if not self.request_history:
            return {'message': 'No data available'}
        
        # Calculate model performance metrics
        model_stats = {}
        
        for entry in self.request_history:
            model = entry['model_used']
            if model not in model_stats:
                model_stats[model] = {
                    'count': 0,
                    'total_confidence': 0.0,
                    'total_distance': 0.0,
                    'total_time': 0.0
                }
            
            stats = model_stats[model]
            stats['count'] += 1
            stats['total_confidence'] += entry['performance']['confidence']
            stats['total_distance'] += entry['performance']['distance']
            stats['total_time'] += entry['performance']['time']
        
        # Calculate averages
        for model, stats in model_stats.items():
            if stats['count'] > 0:
                stats['avg_confidence'] = stats['total_confidence'] / stats['count']
                stats['avg_distance'] = stats['total_distance'] / stats['count']
                stats['avg_time'] = stats['total_time'] / stats['count']
        
        return {
            'total_requests': len(self.request_history),
            'model_performance': model_stats,
            'recent_requests': len([r for r in self.request_history if 
                                  (datetime.now() - r['timestamp']).total_seconds() < 3600])
        }
    
    def _heavy_models_available(self) -> bool:
        """Check if heavy AI models are available and loaded"""
        return (self.transformer_model is not None and 
                self.rl_agent is not None and 
                self.genetic_optimizer is not None)
    
    async def get_status(self) -> Dict[str, Any]:
        """Get AI Engine status"""
        return {
            'initialized': self.is_initialized,
            'models_available': {
                'fast_engine': True,
                'transformer': self.transformer_model is not None,
                'rl_agent': self.rl_agent is not None,
                'genetic': self.genetic_optimizer is not None
            },
            'fast_mode_enabled': self.use_fast_mode,
            'request_history_size': len(self.request_history),
            'cache_stats': traffic_service.get_cache_stats()
        }
    
    async def cleanup(self):
        """Cleanup resources"""
        logging.info("Cleaning up AI Engine...")
        # Clear caches
        self.model_cache.clear()
        traffic_service.clear_cache()
        logging.info("AI Engine cleanup completed")

# Note: AIEngine instance is created and managed in app.main to avoid duplicate instances
