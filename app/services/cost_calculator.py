"""
Enhanced Cost Calculator Service for Indian Market
Realistic fuel, toll, and maintenance costs in Indian Rupees
"""

from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from enum import Enum
import logging


class VehicleType(Enum):
    """Vehicle types with Indian market specifications"""
    BIKE = "bike"
    CAR_PETROL = "car_petrol" 
    CAR_DIESEL = "car_diesel"
    CAR_CNG = "car_cng"
    AUTO_RICKSHAW = "auto_rickshaw"
    BUS = "bus"
    TRUCK = "truck"


@dataclass
class VehicleSpecs:
    """Vehicle specifications for cost calculation"""
    mileage_kmpl: float  # Kilometers per liter
    fuel_price_per_liter_inr: float  # Current fuel price in INR
    maintenance_per_km_inr: float  # Maintenance cost per km
    base_speed_kmph: float  # Average speed
    toll_category: str  # For toll calculations


class CostCalculatorService:
    """Enhanced cost calculator with realistic Indian pricing"""
    
    def __init__(self):
        # Current fuel prices in India (as of October 2024)
        self.fuel_prices = {
            "petrol": 105.0,  # ₹105 per liter (average)
            "diesel": 95.0,   # ₹95 per liter (average)
            "cng": 80.0,      # ₹80 per kg (equivalent)
        }
        
        # Vehicle specifications (realistic Indian data)
        self.vehicle_specs = {
            VehicleType.BIKE: VehicleSpecs(
                mileage_kmpl=45.0,
                fuel_price_per_liter_inr=self.fuel_prices["petrol"],
                maintenance_per_km_inr=0.5,
                base_speed_kmph=40.0,
                toll_category="two_wheeler"
            ),
            VehicleType.CAR_PETROL: VehicleSpecs(
                mileage_kmpl=15.0,
                fuel_price_per_liter_inr=self.fuel_prices["petrol"],
                maintenance_per_km_inr=2.0,
                base_speed_kmph=60.0,
                toll_category="car"
            ),
            VehicleType.CAR_DIESEL: VehicleSpecs(
                mileage_kmpl=20.0,
                fuel_price_per_liter_inr=self.fuel_prices["diesel"],
                maintenance_per_km_inr=2.5,
                base_speed_kmph=60.0,
                toll_category="car"
            ),
            VehicleType.CAR_CNG: VehicleSpecs(
                mileage_kmpl=25.0,  # km per kg for CNG
                fuel_price_per_liter_inr=self.fuel_prices["cng"],
                maintenance_per_km_inr=1.5,
                base_speed_kmph=55.0,
                toll_category="car"
            ),
            VehicleType.AUTO_RICKSHAW: VehicleSpecs(
                mileage_kmpl=35.0,
                fuel_price_per_liter_inr=self.fuel_prices["cng"],
                maintenance_per_km_inr=1.0,
                base_speed_kmph=25.0,
                toll_category="three_wheeler"
            ),
            VehicleType.BUS: VehicleSpecs(
                mileage_kmpl=4.0,
                fuel_price_per_liter_inr=self.fuel_prices["diesel"],
                maintenance_per_km_inr=8.0,
                base_speed_kmph=45.0,
                toll_category="bus"
            ),
            VehicleType.TRUCK: VehicleSpecs(
                mileage_kmpl=3.5,
                fuel_price_per_liter_inr=self.fuel_prices["diesel"],
                maintenance_per_km_inr=12.0,
                base_speed_kmph=50.0,
                toll_category="truck"
            )
        }
        
        # Indian toll rates (per km on average)
        self.toll_rates_per_km = {
            "two_wheeler": 1.0,   # ₹1 per km
            "three_wheeler": 1.5, # ₹1.5 per km
            "car": 2.5,           # ₹2.5 per km
            "bus": 4.0,           # ₹4 per km
            "truck": 6.0          # ₹6 per km
        }
    
    def calculate_route_cost(self, 
                           distance_km: float,
                           vehicle_type: str = "car_petrol",
                           include_tolls: bool = True,
                           route_type: str = "highway") -> Dict[str, float]:
        """
        Calculate comprehensive route cost in Indian Rupees
        
        Args:
            distance_km: Total distance in kilometers
            vehicle_type: Type of vehicle (default: car_petrol)
            include_tolls: Whether to include toll costs
            route_type: highway, city, or mixed
            
        Returns:
            Detailed cost breakdown in INR
        """
        try:
            # Get vehicle specifications
            vehicle_enum = VehicleType(vehicle_type)
            specs = self.vehicle_specs[vehicle_enum]
            
            # Calculate fuel cost
            fuel_cost = (distance_km / specs.mileage_kmpl) * specs.fuel_price_per_liter_inr
            
            # Calculate maintenance cost
            maintenance_cost = distance_km * specs.maintenance_per_km_inr
            
            # Calculate toll cost
            toll_cost = 0.0
            if include_tolls and route_type in ["highway", "mixed"]:
                toll_multiplier = 1.0 if route_type == "highway" else 0.6  # Less tolls in mixed routes
                toll_cost = distance_km * self.toll_rates_per_km[specs.toll_category] * toll_multiplier
            
            # Calculate time-based costs (parking, driver cost for long routes)
            time_hours = distance_km / specs.base_speed_kmph
            time_cost = 0.0
            
            if time_hours > 2:  # Long routes
                if vehicle_enum in [VehicleType.CAR_PETROL, VehicleType.CAR_DIESEL, VehicleType.CAR_CNG]:
                    time_cost = max(0, (time_hours - 2) * 50)  # ₹50/hour for parking/misc
                elif vehicle_enum == VehicleType.TRUCK:
                    time_cost = time_hours * 100  # Driver cost ₹100/hour
            
            # Total cost
            total_cost = fuel_cost + maintenance_cost + toll_cost + time_cost
            
            return {
                "total_cost_inr": round(total_cost, 2),
                "fuel_cost_inr": round(fuel_cost, 2),
                "maintenance_cost_inr": round(maintenance_cost, 2),
                "toll_cost_inr": round(toll_cost, 2),
                "time_cost_inr": round(time_cost, 2),
                "cost_per_km_inr": round(total_cost / max(distance_km, 1), 2),
                "vehicle_type": vehicle_type,
                "distance_km": distance_km
            }
            
        except Exception as e:
            logging.error(f"Cost calculation error: {e}")
            # Fallback calculation
            fallback_cost = distance_km * 8.5  # ₹8.5 per km average
            return {
                "total_cost_inr": round(fallback_cost, 2),
                "fuel_cost_inr": round(fallback_cost * 0.6, 2),
                "maintenance_cost_inr": round(fallback_cost * 0.2, 2),
                "toll_cost_inr": round(fallback_cost * 0.2, 2),
                "time_cost_inr": 0.0,
                "cost_per_km_inr": 8.5,
                "vehicle_type": vehicle_type,
                "distance_km": distance_km
            }
    
    def get_cost_comparison(self, distance_km: float) -> Dict[str, Dict[str, float]]:
        """
        Get cost comparison across different vehicle types
        
        Args:
            distance_km: Distance in kilometers
            
        Returns:
            Cost comparison for all vehicle types
        """
        comparison = {}
        
        for vehicle_type in ["bike", "car_petrol", "car_diesel", "car_cng", "auto_rickshaw"]:
            cost_data = self.calculate_route_cost(distance_km, vehicle_type)
            comparison[vehicle_type] = {
                "total_cost_inr": cost_data["total_cost_inr"],
                "cost_per_km_inr": cost_data["cost_per_km_inr"],
                "fuel_efficiency": self.vehicle_specs[VehicleType(vehicle_type)].mileage_kmpl
            }
        
        return comparison
    
    def get_cost_savings_suggestions(self, distance_km: float, current_vehicle: str = "car_petrol") -> List[Dict[str, any]]:
        """
        Get suggestions for cost savings
        
        Args:
            distance_km: Distance in kilometers
            current_vehicle: Current vehicle type
            
        Returns:
            List of cost-saving suggestions
        """
        current_cost = self.calculate_route_cost(distance_km, current_vehicle)
        all_costs = self.get_cost_comparison(distance_km)
        
        suggestions = []
        
        for vehicle, cost_info in all_costs.items():
            if vehicle != current_vehicle and cost_info["total_cost_inr"] < current_cost["total_cost_inr"]:
                savings = current_cost["total_cost_inr"] - cost_info["total_cost_inr"]
                suggestions.append({
                    "vehicle_type": vehicle,
                    "total_cost_inr": cost_info["total_cost_inr"],
                    "savings_inr": round(savings, 2),
                    "savings_percentage": round((savings / current_cost["total_cost_inr"]) * 100, 1),
                    "recommendation": f"Switch to {vehicle.replace('_', ' ').title()} to save ₹{savings:.0f}"
                })
        
        # Sort by savings amount
        suggestions.sort(key=lambda x: x["savings_inr"], reverse=True)
        return suggestions[:3]  # Top 3 suggestions


# Global service instance
cost_calculator_service = CostCalculatorService()
