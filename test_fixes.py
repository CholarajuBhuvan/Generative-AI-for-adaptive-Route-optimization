"""
Test script to verify all the fixes work correctly
"""

import asyncio
import aiohttp
import json

async def test_fixes():
    """Test the fixes for route generation and location suggestions"""
    
    # Test 1: Location suggestions
    print("🔍 Testing location suggestions...")
    try:
        async with aiohttp.ClientSession() as session:
            # Test location suggestions
            async with session.get("http://localhost:8000/api/v1/location-suggestions?query=Mumbai&limit=5") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Location suggestions work! Found {len(data.get('suggestions', []))} suggestions for 'Mumbai'")
                    for suggestion in data.get('suggestions', [])[:3]:
                        print(f"   - {suggestion['name']}: {suggestion['display_name']}")
                else:
                    print(f"❌ Location suggestions failed: {resp.status}")
    except Exception as e:
        print(f"❌ Error testing location suggestions: {e}")
    
    # Test 2: Popular destinations
    print("\n🌟 Testing popular destinations...")
    try:
        async with aiohttp.ClientSession() as session:
            async with session.get("http://localhost:8000/api/v1/popular-destinations?country=IN") as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print(f"✅ Popular destinations work! Found {len(data.get('destinations', []))} destinations")
                    for dest in data.get('destinations', [])[:5]:
                        print(f"   - {dest['name']}: {dest['display_name']}")
                else:
                    print(f"❌ Popular destinations failed: {resp.status}")
    except Exception as e:
        print(f"❌ Error testing popular destinations: {e}")
    
    # Test 3: Route optimization with location names
    print("\n🛣️ Testing route optimization with location names...")
    try:
        async with aiohttp.ClientSession() as session:
            route_data = {
                "start_location_name": "New Delhi",
                "end_location_name": "Mumbai",
                "travel_mode": "driving",
                "vehicle_type": "average",
                "constraints": {
                    "max_time": 180,
                    "max_distance": 500
                },
                "user_preferences": {
                    "time_weight": 0.4,
                    "distance_weight": 0.3,
                    "cost_weight": 0.2,
                    "traffic_weight": 0.1
                },
                "user_id": "test_user"
            }
            
            async with session.post("http://localhost:8000/api/v1/optimize-route", 
                                  json=route_data,
                                  headers={'Content-Type': 'application/json'}) as resp:
                if resp.status == 200:
                    data = await resp.json()
                    print("✅ Route optimization works!")
                    print(f"   - Route ID: {data.get('route_id')}")
                    print(f"   - Distance: {data.get('total_distance_km', 0):.1f} km")
                    print(f"   - Time: {data.get('total_time_minutes', 0):.0f} minutes")
                    print(f"   - Cost: ₹{data.get('total_cost_inr', 0):.2f}")
                    print(f"   - AI Model: {data.get('ai_model_used')}")
                    print(f"   - Coordinates: {len(data.get('coordinates', []))} waypoints")
                    
                    if data.get('carbon_footprint'):
                        cf = data['carbon_footprint']
                        print(f"   - Carbon: {cf.get('primary_emission', {}).get('total_co2_kg', 0):.2f} kg CO₂")
                else:
                    print(f"❌ Route optimization failed: {resp.status}")
                    error_text = await resp.text()
                    print(f"   Error: {error_text}")
    except Exception as e:
        print(f"❌ Error testing route optimization: {e}")
    
    print("\n🎉 Tests completed!")

if __name__ == "__main__":
    print("🧪 Running Enhanced System Tests...")
    print("=" * 50)
    asyncio.run(test_fixes())
