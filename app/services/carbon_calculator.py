"""
Carbon Footprint Calculator for Route Optimization
Calculates environmental impact of different travel modes
"""

from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from datetime import datetime
import logging


@dataclass
class CarbonEmission:
    """Carbon emission data for a route"""
    total_co2_kg: float
    co2_per_km: float
    travel_mode: str
    distance_km: float
    fuel_consumption_liters: Optional[float] = None
    trees_to_offset: int = 0
    comparison_to_average: float = 0.0
    eco_score: float = 0.0  # 0-100, higher is better


@dataclass
class VehicleEmissionProfile:
    """Emission profile for different vehicle types"""
    vehicle_type: str
    co2_per_km: float  # kg CO2 per km
    fuel_efficiency: float  # km per liter
    fuel_type: str


class CarbonCalculator:
    """
    Calculate carbon footprint for different travel modes
    Based on real-world emission factors
    """
    
    # Emission factors (kg CO2 per km)
    EMISSION_FACTORS = {
        'driving': {
            'petrol_car_small': 0.12,      # Small petrol car
            'petrol_car_medium': 0.17,     # Medium petrol car
            'petrol_car_large': 0.25,      # Large petrol car/SUV
            'diesel_car_small': 0.11,      # Small diesel car
            'diesel_car_medium': 0.15,     # Medium diesel car
            'diesel_car_large': 0.22,      # Large diesel car
            'electric_car': 0.05,          # Electric car (grid average)
            'hybrid_car': 0.10,            # Hybrid car
            'motorcycle': 0.08,            # Motorcycle
            'scooter': 0.06,               # Scooter
            'average': 0.17                # Average car
        },
        'transit': {
            'bus': 0.08,                   # Public bus per passenger
            'metro': 0.04,                 # Metro/subway per passenger
            'train': 0.04,                 # Electric train per passenger
            'tram': 0.03,                  # Tram per passenger
            'average': 0.05
        },
        'cycling': 0.0,                    # Zero emissions
        'walking': 0.0,                    # Zero emissions
        'flying': 0.25,                    # Air travel per passenger
        'taxi': 0.20,                      # Taxi/ride-sharing
        'auto_rickshaw': 0.09              # Auto-rickshaw (India specific)
    }
    
    # Fuel efficiency (km per liter)
    FUEL_EFFICIENCY = {
        'petrol_car_small': 18,
        'petrol_car_medium': 14,
        'petrol_car_large': 10,
        'diesel_car_small': 22,
        'diesel_car_medium': 18,
        'diesel_car_large': 14,
        'hybrid_car': 25,
        'motorcycle': 40,
        'scooter': 45,
        'average': 15
    }
    
    # Trees needed to offset 1 ton of CO2 per year
    TREES_PER_TON_CO2 = 50
    
    def __init__(self):
        self.vehicle_profiles = self._initialize_vehicle_profiles()
    
    def _initialize_vehicle_profiles(self) -> Dict[str, VehicleEmissionProfile]:
        """Initialize vehicle emission profiles"""
        profiles = {}
        
        for vehicle_type, co2_per_km in self.EMISSION_FACTORS['driving'].items():
            if vehicle_type != 'average':
                fuel_type = 'petrol' if 'petrol' in vehicle_type else \
                           'diesel' if 'diesel' in vehicle_type else \
                           'electric' if 'electric' in vehicle_type else \
                           'hybrid' if 'hybrid' in vehicle_type else 'petrol'
                
                profiles[vehicle_type] = VehicleEmissionProfile(
                    vehicle_type=vehicle_type,
                    co2_per_km=co2_per_km,
                    fuel_efficiency=self.FUEL_EFFICIENCY.get(vehicle_type, 15),
                    fuel_type=fuel_type
                )
        
        return profiles
    
    def calculate_emissions(self, 
                          distance_km: float, 
                          travel_mode: str,
                          vehicle_type: Optional[str] = None) -> CarbonEmission:
        """
        Calculate carbon emissions for a route
        
        Args:
            distance_km: Distance in kilometers
            travel_mode: Mode of travel (driving, walking, cycling, transit)
            vehicle_type: Specific vehicle type (optional)
            
        Returns:
            Carbon emission data
        """
        # Get emission factor
        co2_per_km = self._get_emission_factor(travel_mode, vehicle_type)
        
        # Calculate total emissions
        total_co2_kg = distance_km * co2_per_km
        
        # Calculate fuel consumption if applicable
        fuel_consumption = None
        if travel_mode == 'driving' and vehicle_type:
            fuel_efficiency = self.FUEL_EFFICIENCY.get(vehicle_type, 15)
            fuel_consumption = distance_km / fuel_efficiency
        
        # Calculate trees needed to offset
        trees_to_offset = int((total_co2_kg / 1000) * self.TREES_PER_TON_CO2)
        
        # Calculate comparison to average
        average_co2 = distance_km * self.EMISSION_FACTORS['driving']['average']
        comparison = ((total_co2_kg - average_co2) / average_co2 * 100) if average_co2 > 0 else 0
        
        # Calculate eco score (0-100, higher is better)
        eco_score = self._calculate_eco_score(co2_per_km, travel_mode)
        
        return CarbonEmission(
            total_co2_kg=round(total_co2_kg, 3),
            co2_per_km=round(co2_per_km, 3),
            travel_mode=travel_mode,
            distance_km=distance_km,
            fuel_consumption_liters=round(fuel_consumption, 2) if fuel_consumption else None,
            trees_to_offset=trees_to_offset,
            comparison_to_average=round(comparison, 1),
            eco_score=round(eco_score, 1)
        )
    
    def compare_travel_modes(self, distance_km: float) -> Dict[str, CarbonEmission]:
        """
        Compare carbon emissions across different travel modes
        
        Args:
            distance_km: Distance in kilometers
            
        Returns:
            Dictionary of emissions for each travel mode
        """
        comparisons = {}
        
        # Driving modes
        comparisons['car_average'] = self.calculate_emissions(distance_km, 'driving', 'average')
        comparisons['car_electric'] = self.calculate_emissions(distance_km, 'driving', 'electric_car')
        comparisons['car_hybrid'] = self.calculate_emissions(distance_km, 'driving', 'hybrid_car')
        
        # Public transit
        comparisons['bus'] = self.calculate_emissions(distance_km, 'transit', 'bus')
        comparisons['metro'] = self.calculate_emissions(distance_km, 'transit', 'metro')
        
        # Eco-friendly
        comparisons['cycling'] = self.calculate_emissions(distance_km, 'cycling')
        comparisons['walking'] = self.calculate_emissions(distance_km, 'walking')
        
        # India-specific
        comparisons['auto_rickshaw'] = self.calculate_emissions(
            distance_km, 'driving', None
        )
        comparisons['auto_rickshaw'].co2_per_km = self.EMISSION_FACTORS['auto_rickshaw']
        comparisons['auto_rickshaw'].total_co2_kg = distance_km * self.EMISSION_FACTORS['auto_rickshaw']
        
        return comparisons
    
    def get_eco_recommendations(self, 
                               distance_km: float,
                               current_mode: str,
                               current_vehicle: Optional[str] = None) -> List[Dict[str, Any]]:
        """
        Get recommendations for reducing carbon footprint
        
        Args:
            distance_km: Distance in kilometers
            current_mode: Current travel mode
            current_vehicle: Current vehicle type
            
        Returns:
            List of recommendations
        """
        current_emission = self.calculate_emissions(distance_km, current_mode, current_vehicle)
        all_modes = self.compare_travel_modes(distance_km)
        
        recommendations = []
        
        # Find better alternatives
        for mode, emission in all_modes.items():
            if emission.total_co2_kg < current_emission.total_co2_kg:
                savings_kg = current_emission.total_co2_kg - emission.total_co2_kg
                savings_percent = (savings_kg / current_emission.total_co2_kg) * 100
                
                recommendations.append({
                    'mode': mode,
                    'savings_kg': round(savings_kg, 2),
                    'savings_percent': round(savings_percent, 1),
                    'new_total_kg': round(emission.total_co2_kg, 2),
                    'eco_score': emission.eco_score,
                    'description': self._get_mode_description(mode)
                })
        
        # Sort by savings
        recommendations.sort(key=lambda x: x['savings_kg'], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def calculate_route_environmental_impact(self,
                                            distance_km: float,
                                            time_minutes: float,
                                            travel_mode: str,
                                            vehicle_type: Optional[str] = None) -> Dict[str, Any]:
        """
        Calculate comprehensive environmental impact of a route
        
        Args:
            distance_km: Distance in kilometers
            time_minutes: Travel time in minutes
            travel_mode: Mode of travel
            vehicle_type: Specific vehicle type
            
        Returns:
            Comprehensive environmental impact analysis
        """
        emission = self.calculate_emissions(distance_km, travel_mode, vehicle_type)
        alternatives = self.compare_travel_modes(distance_km)
        recommendations = self.get_eco_recommendations(distance_km, travel_mode, vehicle_type)
        
        # Calculate additional metrics
        fuel_cost_inr = 0.0
        if emission.fuel_consumption_liters:
            # Average fuel price in India (₹100 per liter for petrol)
            fuel_price_per_liter = 100.0
            fuel_cost_inr = emission.fuel_consumption_liters * fuel_price_per_liter
        
        # Calculate air quality impact
        air_quality_impact = self._calculate_air_quality_impact(emission.total_co2_kg, distance_km)
        
        return {
            'primary_emission': {
                'total_co2_kg': emission.total_co2_kg,
                'co2_per_km': emission.co2_per_km,
                'trees_to_offset': emission.trees_to_offset,
                'eco_score': emission.eco_score,
                'fuel_consumption_liters': emission.fuel_consumption_liters,
                'fuel_cost_inr': round(fuel_cost_inr, 2)
            },
            'comparison': {
                'vs_average_car': emission.comparison_to_average,
                'rank': self._rank_emission(emission, alternatives)
            },
            'alternatives': {
                mode: {
                    'co2_kg': alt.total_co2_kg,
                    'eco_score': alt.eco_score,
                    'savings_vs_current': round(emission.total_co2_kg - alt.total_co2_kg, 2)
                }
                for mode, alt in alternatives.items()
            },
            'recommendations': recommendations,
            'air_quality_impact': air_quality_impact,
            'environmental_metrics': {
                'carbon_intensity': round(emission.total_co2_kg / time_minutes, 3) if time_minutes > 0 else 0,
                'efficiency_score': round((distance_km / emission.total_co2_kg) * 10, 1) if emission.total_co2_kg > 0 else 100,
                'sustainability_rating': self._get_sustainability_rating(emission.eco_score)
            }
        }
    
    def _get_emission_factor(self, travel_mode: str, vehicle_type: Optional[str] = None) -> float:
        """Get emission factor for a travel mode"""
        if travel_mode == 'driving':
            if vehicle_type and vehicle_type in self.EMISSION_FACTORS['driving']:
                return self.EMISSION_FACTORS['driving'][vehicle_type]
            return self.EMISSION_FACTORS['driving']['average']
        elif travel_mode == 'transit':
            if vehicle_type and vehicle_type in self.EMISSION_FACTORS['transit']:
                return self.EMISSION_FACTORS['transit'][vehicle_type]
            return self.EMISSION_FACTORS['transit']['average']
        elif travel_mode in self.EMISSION_FACTORS:
            return self.EMISSION_FACTORS[travel_mode]
        else:
            return self.EMISSION_FACTORS['driving']['average']
    
    def _calculate_eco_score(self, co2_per_km: float, travel_mode: str) -> float:
        """Calculate eco score (0-100, higher is better)"""
        if travel_mode in ['walking', 'cycling']:
            return 100.0
        
        # Maximum reasonable emission (large SUV)
        max_emission = 0.30
        
        # Score inversely proportional to emissions
        score = max(0, 100 * (1 - (co2_per_km / max_emission)))
        
        return score
    
    def _calculate_air_quality_impact(self, total_co2_kg: float, distance_km: float) -> Dict[str, Any]:
        """Calculate impact on air quality"""
        # Estimate other pollutants based on CO2
        # These are rough estimates
        nox_kg = total_co2_kg * 0.015  # NOx emissions
        pm25_g = total_co2_kg * 0.5    # PM2.5 emissions in grams
        
        return {
            'nox_emissions_kg': round(nox_kg, 3),
            'pm25_emissions_g': round(pm25_g, 2),
            'air_quality_index_impact': round(pm25_g / distance_km, 2) if distance_km > 0 else 0
        }
    
    def _rank_emission(self, current: CarbonEmission, alternatives: Dict[str, CarbonEmission]) -> str:
        """Rank current emission among alternatives"""
        all_emissions = [alt.total_co2_kg for alt in alternatives.values()]
        all_emissions.append(current.total_co2_kg)
        all_emissions.sort()
        
        rank = all_emissions.index(current.total_co2_kg) + 1
        total = len(all_emissions)
        
        if rank == 1:
            return f"Best (1/{total})"
        elif rank <= total * 0.33:
            return f"Good ({rank}/{total})"
        elif rank <= total * 0.66:
            return f"Average ({rank}/{total})"
        else:
            return f"Poor ({rank}/{total})"
    
    def _get_mode_description(self, mode: str) -> str:
        """Get description for a travel mode"""
        descriptions = {
            'car_average': 'Average petrol/diesel car',
            'car_electric': 'Electric vehicle',
            'car_hybrid': 'Hybrid vehicle',
            'bus': 'Public bus',
            'metro': 'Metro/subway',
            'cycling': 'Bicycle',
            'walking': 'Walking',
            'auto_rickshaw': 'Auto-rickshaw'
        }
        return descriptions.get(mode, mode.replace('_', ' ').title())
    
    def _get_sustainability_rating(self, eco_score: float) -> str:
        """Get sustainability rating based on eco score"""
        if eco_score >= 90:
            return 'Excellent'
        elif eco_score >= 70:
            return 'Good'
        elif eco_score >= 50:
            return 'Fair'
        elif eco_score >= 30:
            return 'Poor'
        else:
            return 'Very Poor'


# Global carbon calculator instance
carbon_calculator = CarbonCalculator()
