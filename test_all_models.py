#!/usr/bin/env python3
"""
Test script to verify all three AI models are working
"""

import requests
import json
import time
from typing import Dict, List

BASE_URL = "http://127.0.0.1:8000/api/v1"

def test_route_optimization(payload: Dict) -> Dict:
    """Test route optimization API"""
    try:
        response = requests.post(f"{BASE_URL}/optimize-route", json=payload)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        print(f"Error testing route: {e}")
        return {}

def main():
    print("🚀 Testing AI Route Optimization - All Three Models")
    print("=" * 60)
    
    test_scenarios = [
        {
            "name": "Short Distance (Should use Transformer)",
            "payload": {
                "start_lat": 28.6139,
                "start_lng": 77.2090,
                "end_lat": 28.6200,
                "end_lng": 77.2100,
                "travel_mode": "driving",
                "constraints": {}
            }
        },
        {
            "name": "High Traffic Scenario (Should use RL Agent)",
            "payload": {
                "start_lat": 28.6545,
                "start_lng": 77.2442,
                "end_lat": 28.6304,
                "end_lng": 77.2177,
                "travel_mode": "driving",
                "user_preferences": {"time_weight": 0.8, "avoid_traffic": 0.9}
            }
        },
        {
            "name": "Complex Constraints (Should use Genetic)",
            "payload": {
                "start_lat": 28.6139,
                "start_lng": 77.2090,
                "end_lat": 28.5355,
                "end_lng": 77.3910,
                "travel_mode": "driving",
                "constraints": {
                    "avoid_tolls": True,
                    "prefer_highways": False,
                    "max_time": 60,
                    "prefer_scenic": True
                }
            }
        },
        {
            "name": "Long Distance Route (Should use Genetic)",
            "payload": {
                "start_lat": 28.6139,
                "start_lng": 77.2090,
                "end_lat": 19.0760,
                "end_lng": 72.8777,
                "travel_mode": "driving",
                "user_preferences": {"distance_weight": 0.4, "time_weight": 0.6}
            }
        },
        {
            "name": "Time-based Rotation Test 1",
            "payload": {
                "start_lat": 28.6139,
                "start_lng": 77.2090,
                "end_lat": 28.6250,
                "end_lng": 77.2150,
                "travel_mode": "driving"
            }
        }
    ]
    
    results = []
    
    for i, scenario in enumerate(test_scenarios):
        print(f"\n🧪 Test {i+1}: {scenario['name']}")
        print("-" * 40)
        
        # Add small delay between requests for time-based rotation
        if i > 0:
            time.sleep(2)
        
        result = test_route_optimization(scenario['payload'])
        
        if result:
            ai_model = result.get('ai_model_used', 'Unknown')
            confidence = result.get('confidence_score', 0)
            distance = result.get('total_distance_km', 0)
            time_min = result.get('total_time_minutes', 0)
            
            print(f"✅ AI Model Used: {ai_model}")
            print(f"📊 Confidence: {confidence:.3f}")
            print(f"📏 Distance: {distance:.2f} km")
            print(f"⏱️  Time: {time_min:.1f} minutes")
            
            results.append({
                'scenario': scenario['name'],
                'model': ai_model,
                'confidence': confidence,
                'distance': distance,
                'time': time_min
            })
        else:
            print("❌ Failed to get result")
    
    # Summary
    print("\n" + "="*60)
    print("📋 SUMMARY - AI Models Used:")
    print("="*60)
    
    models_used = {}
    for result in results:
        model = result['model']
        if model in models_used:
            models_used[model] += 1
        else:
            models_used[model] = 1
    
    for model, count in models_used.items():
        print(f"🤖 {model}: {count} request(s)")
    
    print(f"\n✅ Total Models Active: {len(models_used)}/3")
    
    if len(models_used) >= 2:
        print("🎉 SUCCESS: Multiple AI models are working!")
    else:
        print("⚠️  WARNING: Only one model type detected")
    
    # Test analytics endpoint
    print("\n" + "="*60)
    print("📈 ANALYTICS CHECK:")
    print("="*60)
    
    try:
        analytics_response = requests.get(f"{BASE_URL}/analytics")
        analytics_response.raise_for_status()
        analytics = analytics_response.json()
        
        ai_engine_stats = analytics.get('ai_engine', {})
        total_requests = ai_engine_stats.get('total_requests', 0)
        model_performance = ai_engine_stats.get('model_performance', {})
        
        print(f"📊 Total Requests Processed: {total_requests}")
        print("📋 Model Performance:")
        
        for model, stats in model_performance.items():
            count = stats.get('count', 0)
            avg_confidence = stats.get('avg_confidence', 0)
            print(f"   🤖 {model}: {count} requests, avg confidence: {avg_confidence:.3f}")
            
    except Exception as e:
        print(f"❌ Error fetching analytics: {e}")

if __name__ == "__main__":
    main()
