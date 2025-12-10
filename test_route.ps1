$body = @{
    start_lat = 28.6139
    start_lng = 77.2090
    end_lat = 28.5355
    end_lng = 77.3910
    travel_mode = "driving"
} | ConvertTo-Json

$response = Invoke-RestMethod -Uri "http://localhost:8000/api/v1/optimize-route" -Method POST -ContentType "application/json" -Body $body

Write-Host "Route ID: $($response.route_id)"
Write-Host "Distance: $($response.total_distance_km) km"
Write-Host "Time: $($response.total_time_minutes) minutes"
Write-Host "Cost: ₹$($response.total_cost_inr)"
Write-Host "Confidence: $($response.confidence_score)"
Write-Host "Route Points: $($response.coordinates.Count)"
Write-Host "Generation Time: $([math]::Round($response.metadata.generation_time_ms))ms"
