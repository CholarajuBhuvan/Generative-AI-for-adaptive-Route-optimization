"""
Optimized Genetic Algorithm for Route Optimization
Generates more direct, Google Maps-like routes
"""

import random
import numpy as np
from typing import List, Dict, Tuple, Optional, Callable
import json
from dataclasses import dataclass
from pathlib import Path
import copy


@dataclass
class RouteGene:
    """Represents a single route as a gene in the genetic algorithm"""
    waypoints: List[Tuple[float, float]]  # List of (lat, lng) coordinates
    fitness_score: float = 0.0
    objectives: Dict[str, float] = None  # Individual objective scores
    
    def __post_init__(self):
        if self.objectives is None:
            self.objectives = {}


@dataclass
class OptimizationObjectives:
    """Defines the objectives for route optimization"""
    minimize_time: bool = True
    minimize_distance: bool = True
    minimize_cost: bool = False
    maximize_scenic_value: bool = False
    minimize_traffic: bool = False
    
    time_weight: float = 0.3
    distance_weight: float = 0.7  # Prioritize distance
    cost_weight: float = 0.0
    scenic_weight: float = 0.0
    traffic_weight: float = 0.0


class OptimizedGeneticRouteOptimizer:
    """
    Optimized Genetic Algorithm for route optimization
    Generates more direct, Google Maps-like routes
    """
    
    def __init__(self, 
                 population_size: int = 50,
                 max_generations: int = 30,
                 mutation_rate: float = 0.05,
                 crossover_rate: float = 0.8,
                 elitism_rate: float = 0.1,
                 objectives: OptimizationObjectives = None):
        
        self.population_size = population_size
        self.max_generations = max_generations
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism_rate = elitism_rate
        self.objectives = objectives or OptimizationObjectives()
        
        # Population storage
        self.population: List[RouteGene] = []
        self.generation = 0
        self.best_solution: Optional[RouteGene] = None
        self.evolution_history: List[Dict] = []
        
        # Fitness evaluation functions
        self.fitness_functions: Dict[str, Callable] = {
            'time': self._evaluate_time_objective,
            'distance': self._evaluate_distance_objective,
            'cost': self._evaluate_cost_objective,
            'traffic': self._evaluate_traffic_objective
        }
    
    def optimize_route(self, 
                      start: Tuple[float, float],
                      end: Tuple[float, float],
                      constraints: Dict,
                      traffic_data: Dict,
                      map_data: Dict) -> Dict:
        """
        Optimize route using genetic algorithm
        """
        # Initialize population with more direct routes
        self._initialize_population(start, end, constraints, map_data)
        
        # Evolution loop
        for generation in range(self.max_generations):
            self.generation = generation
            
            # Evaluate fitness
            self._evaluate_population(traffic_data, map_data)
            
            # Track best solution
            generation_best = max(self.population, key=lambda x: x.fitness_score)
            if self.best_solution is None or generation_best.fitness_score > self.best_solution.fitness_score:
                self.best_solution = copy.deepcopy(generation_best)
            
            # Record evolution statistics
            self._record_generation_stats()
            
            # Create next generation
            self._evolve_population()
            
            # Log progress
            if generation % 10 == 0:
                print(f"Generation {generation}: Best Fitness = {self.best_solution.fitness_score:.4f}")
        
        # Final evaluation
        self._evaluate_population(traffic_data, map_data)
        self.best_solution = max(self.population, key=lambda x: x.fitness_score)
        
        return self._generate_results()
    
    def _initialize_population(self, 
                              start: Tuple[float, float],
                              end: Tuple[float, float],
                              constraints: Dict,
                              map_data: Dict):
        """Initialize population with more direct routes"""
        self.population = []
        
        # Calculate direct distance
        direct_distance = self._calculate_distance(start, end)
        
        for _ in range(self.population_size):
            # Generate more direct waypoints
            waypoints = self._generate_direct_waypoints(start, end, direct_distance)
            gene = RouteGene(waypoints=waypoints)
            self.population.append(gene)
    
    def _generate_direct_waypoints(self, 
                                  start: Tuple[float, float],
                                  end: Tuple[float, float],
                                  direct_distance: float) -> List[Tuple[float, float]]:
        """Generate more direct waypoints similar to Google Maps"""
        waypoints = [start]
        
        # Determine number of waypoints based on distance
        if direct_distance < 1.0:  # Very short distance
            num_waypoints = 2
        elif direct_distance < 5.0:  # Short distance
            num_waypoints = 3
        elif direct_distance < 20.0:  # Medium distance
            num_waypoints = 4
        else:  # Long distance
            num_waypoints = 5
        
        # Generate waypoints along the direct path
        for i in range(1, num_waypoints - 1):
            # Linear interpolation between start and end
            t = i / (num_waypoints - 1)
            
            # Calculate base point
            base_lat = start[0] + t * (end[0] - start[0])
            base_lng = start[1] + t * (end[1] - start[1])
            
            # Add small random variation (much smaller than original)
            lat_offset = random.uniform(-0.0005, 0.0005)  # Very small offset
            lng_offset = random.uniform(-0.0005, 0.0005)
            
            lat = base_lat + lat_offset
            lng = base_lng + lng_offset
            
            # Ensure waypoint is within bounds
            lat = max(-90, min(90, lat))
            lng = max(-180, min(180, lng))
            
            waypoints.append((lat, lng))
        
        waypoints.append(end)
        return waypoints
    
    def _evaluate_population(self, traffic_data: Dict, map_data: Dict):
        """Evaluate fitness for entire population"""
        for gene in self.population:
            gene.objectives = {}
            
            # Evaluate each objective
            for obj_name, func in self.fitness_functions.items():
                gene.objectives[obj_name] = func(gene.waypoints, traffic_data, map_data)
            
            # Calculate weighted fitness score
            gene.fitness_score = self._calculate_fitness_score(gene.objectives)
    
    def _calculate_fitness_score(self, objectives: Dict[str, float]) -> float:
        """Calculate weighted fitness score from individual objectives"""
        score = 0.0
        
        if self.objectives.minimize_time:
            score += self.objectives.time_weight * (1.0 - objectives['time'])
        
        if self.objectives.minimize_distance:
            score += self.objectives.distance_weight * (1.0 - objectives['distance'])
        
        if self.objectives.minimize_cost:
            score += self.objectives.cost_weight * (1.0 - objectives['cost'])
        
        if self.objectives.maximize_scenic_value:
            score += self.objectives.scenic_weight * objectives['scenic']
        
        if self.objectives.minimize_traffic:
            score += self.objectives.traffic_weight * (1.0 - objectives['traffic'])
        
        return score
    
    def _evaluate_time_objective(self, 
                                waypoints: List[Tuple[float, float]],
                                traffic_data: Dict,
                                map_data: Dict) -> float:
        """Evaluate time-based objective (normalized 0-1, lower is better)"""
        total_time = 0.0
        
        for i in range(len(waypoints) - 1):
            segment_time = self._calculate_segment_time(
                waypoints[i], waypoints[i + 1], traffic_data
            )
            total_time += segment_time
        
        # Normalize to 0-1 (assuming max reasonable time is 180 minutes)
        return min(1.0, total_time / 180.0)
    
    def _evaluate_distance_objective(self, 
                                    waypoints: List[Tuple[float, float]],
                                    traffic_data: Dict,
                                    map_data: Dict) -> float:
        """Evaluate distance-based objective (normalized 0-1, lower is better)"""
        total_distance = 0.0
        
        for i in range(len(waypoints) - 1):
            distance = self._calculate_distance(waypoints[i], waypoints[i + 1])
            total_distance += distance
        
        # Normalize to 0-1 (assuming max reasonable distance is 200 km)
        return min(1.0, total_distance / 200.0)
    
    def _evaluate_cost_objective(self, 
                                waypoints: List[Tuple[float, float]],
                                traffic_data: Dict,
                                map_data: Dict) -> float:
        """Evaluate cost-based objective (normalized 0-1, lower is better)"""
        total_cost = 0.0
        
        for i in range(len(waypoints) - 1):
            segment_cost = self._calculate_segment_cost(
                waypoints[i], waypoints[i + 1], map_data
            )
            total_cost += segment_cost
        
        # Normalize to 0-1 (assuming max reasonable cost is )
        return min(1.0, total_cost / 50.0)
    
    def _evaluate_traffic_objective(self, 
                                   waypoints: List[Tuple[float, float]],
                                   traffic_data: Dict,
                                   map_data: Dict) -> float:
        """Evaluate traffic-based objective (normalized 0-1, lower is better)"""
        total_traffic = 0.0
        
        for i in range(len(waypoints) - 1):
            traffic_level = self._get_traffic_level(
                waypoints[i], waypoints[i + 1], traffic_data
            )
            total_traffic += traffic_level
        
        # Average traffic level
        avg_traffic = total_traffic / max(1, len(waypoints) - 1)
        return avg_traffic
    
    def _calculate_distance(self, 
                           start: Tuple[float, float],
                           end: Tuple[float, float]) -> float:
        """Calculate distance between two points using Haversine formula"""
        lat1, lon1 = start
        lat2, lon2 = end
        
        # Convert to radians
        lat1, lon1, lat2, lon2 = map(np.radians, [lat1, lon1, lat2, lon2])
        
        # Haversine formula
        dlat = lat2 - lat1
        dlon = lon2 - lon1
        a = np.sin(dlat/2)**2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        
        # Earth radius in kilometers
        r = 6371
        return c * r
    
    def _calculate_segment_time(self, 
                               start: Tuple[float, float],
                               end: Tuple[float, float],
                               traffic_data: Dict) -> float:
        """Calculate time for a route segment"""
        distance = self._calculate_distance(start, end)
        
        # Realistic base speed based on segment distance (km/h)
        if distance < 10:
            base_speed = 35.0  # Urban/city driving
        elif distance < 50:
            base_speed = 50.0  # Mixed urban-highway
        else:
            base_speed = 65.0  # Highway/long distance
        
        # Adjust for traffic
        traffic_level = self._get_traffic_level(start, end, traffic_data)
        speed = base_speed * (1.0 - traffic_level * 0.5)  # Reduce speed by up to 50%
        
        # Ensure minimum realistic speed
        speed = max(speed, 15.0)
        
        # Calculate time in minutes with 10% overhead for stops and delays
        time_hours = distance / speed
        return time_hours * 60.0 * 1.10
    
    def _calculate_segment_cost(self, 
                               start: Tuple[float, float],
                               end: Tuple[float, float],
                               map_data: Dict) -> float:
        """Calculate cost for a route segment"""
        distance = self._calculate_distance(start, end)
        
        # Base cost per km
        base_cost_per_km = 0.5
        
        # Check for toll roads
        if self._is_toll_road(start, end, map_data):
            base_cost_per_km += 0.3
        
        return distance * base_cost_per_km
    
    def _get_traffic_level(self, 
                          start: Tuple[float, float],
                          end: Tuple[float, float],
                          traffic_data: Dict) -> float:
        """Get traffic level for a route segment"""
        # Simplified traffic model
        mid_lat = (start[0] + end[0]) / 2
        mid_lon = (start[1] + end[1]) / 2
        
        # Generate synthetic traffic based on location
        traffic = 0.2 + 0.3 * np.sin(mid_lat * 10) * np.cos(mid_lon * 10)
        return max(0, min(1, traffic))
    
    def _is_toll_road(self, 
                     start: Tuple[float, float],
                     end: Tuple[float, float],
                     map_data: Dict) -> bool:
        """Check if route segment is a toll road"""
        # Simplified toll road detection
        mid_lat = (start[0] + end[0]) / 2
        mid_lon = (start[1] + end[1]) / 2
        
        # Some areas are toll roads (simplified)
        return 0.1 < mid_lat < 0.3 and 0.1 < mid_lon < 0.3
    
    def _evolve_population(self):
        """Create next generation using genetic operators"""
        new_population = []
        
        # Elitism: keep best individuals
        num_elites = int(self.population_size * self.elitism_rate)
        elites = sorted(self.population, key=lambda x: x.fitness_score, reverse=True)[:num_elites]
        new_population.extend(elites)
        
        # Generate remaining individuals through crossover and mutation
        while len(new_population) < self.population_size:
            # Selection
            parent1 = self._tournament_selection()
            parent2 = self._tournament_selection()
            
            # Crossover
            if random.random() < self.crossover_rate:
                child1, child2 = self._crossover(parent1, parent2)
            else:
                child1, child2 = copy.deepcopy(parent1), copy.deepcopy(parent2)
            
            # Mutation
            if random.random() < self.mutation_rate:
                child1 = self._mutate(child1)
            if random.random() < self.mutation_rate:
                child2 = self._mutate(child2)
            
            new_population.extend([child1, child2])
        
        # Ensure population size
        self.population = new_population[:self.population_size]
    
    def _tournament_selection(self, tournament_size: int = 3) -> RouteGene:
        """Tournament selection for parent selection"""
        tournament = random.sample(self.population, min(tournament_size, len(self.population)))
        return max(tournament, key=lambda x: x.fitness_score)
    
    def _crossover(self, parent1: RouteGene, parent2: RouteGene) -> Tuple[RouteGene, RouteGene]:
        """Crossover operation to create offspring"""
        # Uniform crossover for waypoints
        child1_waypoints = []
        child2_waypoints = []
        
        max_len = max(len(parent1.waypoints), len(parent2.waypoints))
        
        for i in range(max_len):
            if i < len(parent1.waypoints) and i < len(parent2.waypoints):
                if random.random() < 0.5:
                    child1_waypoints.append(parent1.waypoints[i])
                    child2_waypoints.append(parent2.waypoints[i])
                else:
                    child1_waypoints.append(parent2.waypoints[i])
                    child2_waypoints.append(parent1.waypoints[i])
            elif i < len(parent1.waypoints):
                child1_waypoints.append(parent1.waypoints[i])
                child2_waypoints.append(parent1.waypoints[i])
            else:
                child1_waypoints.append(parent2.waypoints[i])
                child2_waypoints.append(parent2.waypoints[i])
        
        return RouteGene(waypoints=child1_waypoints), RouteGene(waypoints=child2_waypoints)
    
    def _mutate(self, gene: RouteGene) -> RouteGene:
        """Mutation operation"""
        mutated_gene = copy.deepcopy(gene)
        
        # Random waypoint mutation
        if len(mutated_gene.waypoints) > 2:
            # Select random waypoint to mutate (excluding start and end)
            idx = random.randint(1, len(mutated_gene.waypoints) - 2)
            
            # Add small random offset
            lat_offset = random.uniform(-0.0005, 0.0005)  # Very small mutation
            lon_offset = random.uniform(-0.0005, 0.0005)
            
            old_lat, old_lng = mutated_gene.waypoints[idx]
            new_lat = max(-90, min(90, old_lat + lat_offset))
            new_lng = max(-180, min(180, old_lng + lon_offset))
            
            mutated_gene.waypoints[idx] = (new_lat, new_lng)
        
        return mutated_gene
    
    def _record_generation_stats(self):
        """Record statistics for current generation"""
        fitness_scores = [gene.fitness_score for gene in self.population]
        
        stats = {
            'generation': self.generation,
            'best_fitness': max(fitness_scores),
            'worst_fitness': min(fitness_scores),
            'avg_fitness': np.mean(fitness_scores),
            'std_fitness': np.std(fitness_scores)
        }
        
        self.evolution_history.append(stats)
    
    def _generate_results(self) -> Dict:
        """Generate final optimization results"""
        if not self.best_solution:
            raise ValueError("No solution found")
        
        # Convert waypoints to route format
        route_coordinates = []
        for i, (lat, lng) in enumerate(self.best_solution.waypoints):
            route_coordinates.append({
                'lat': lat,
                'lng': lng,
                'time': i * 2.0,  # Estimated time at each point
                'confidence': 0.9  # High confidence for direct routes
            })
        
        return {
            'route': {
                'coordinates': route_coordinates,
                'total_distance_km': self._calculate_total_distance(self.best_solution.waypoints),
                'total_time_minutes': self._calculate_total_time(self.best_solution.waypoints),
                'total_cost': self._calculate_total_cost(self.best_solution.waypoints),
                'fitness_score': self.best_solution.fitness_score,
                'objectives': self.best_solution.objectives
            },
            'optimization_stats': {
                'generations': self.max_generations,
                'population_size': self.population_size,
                'final_fitness': self.best_solution.fitness_score,
                'evolution_history': self.evolution_history
            },
            'algorithm_config': {
                'mutation_rate': self.mutation_rate,
                'crossover_rate': self.crossover_rate,
                'elitism_rate': self.elitism_rate,
                'objectives': self.objectives.__dict__
            }
        }
    
    def _calculate_total_distance(self, waypoints: List[Tuple[float, float]]) -> float:
        """Calculate total distance of route"""
        total = 0.0
        for i in range(len(waypoints) - 1):
            total += self._calculate_distance(waypoints[i], waypoints[i + 1])
        return total
    
    def _calculate_total_time(self, waypoints: List[Tuple[float, float]]) -> float:
        """Calculate total time of route"""
        total = 0.0
        for i in range(len(waypoints) - 1):
            total += self._calculate_segment_time(waypoints[i], waypoints[i + 1], {})
        return total
    
    def _calculate_total_cost(self, waypoints: List[Tuple[float, float]]) -> float:
        """Calculate total cost of route"""
        total = 0.0
        for i in range(len(waypoints) - 1):
            total += self._calculate_segment_cost(waypoints[i], waypoints[i + 1], {})
        return total
