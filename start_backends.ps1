$env:Path = [System.Environment]::GetEnvironmentVariable("Path","Machine") + ";" + [System.Environment]::GetEnvironmentVariable("Path","User")

Write-Host "Starting Dummy Backends for Local Testing..." -ForegroundColor Cyan

Start-Process -FilePath "go" -ArgumentList "run ./testutil/dummy_backend.go -port 9001" -WindowStyle Normal
Start-Process -FilePath "go" -ArgumentList "run ./testutil/dummy_backend.go -port 9002" -WindowStyle Normal
Start-Process -FilePath "go" -ArgumentList "run ./testutil/dummy_backend.go -port 9003" -WindowStyle Normal
Start-Process -FilePath "go" -ArgumentList "run ./testutil/dummy_backend.go -port 9004" -WindowStyle Normal
Start-Process -FilePath "go" -ArgumentList "run ./testutil/dummy_backend.go -port 9005" -WindowStyle Normal

Write-Host "Backends are running! You can now test the load balancer in your browser at http://localhost:8080" -ForegroundColor Green
