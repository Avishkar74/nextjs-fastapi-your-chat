#!/bin/bash
echo "🚀 Deploying Backend to Railway..."

# Login to Railway (you'll need to do this once)
echo "Please login to Railway when prompted..."
railway login

# Create new project
echo "Creating new Railway project..."
railway new

# Set Python buildpack
railway variables set RAILWAY_DOCKERFILE_PATH=Dockerfile.railway

# Set environment variables
echo "Setting environment variables..."
echo "You'll need to set these manually in Railway dashboard:"
echo "- SUPABASE_URL"
echo "- SUPABASE_KEY" 
echo "- GEMINI_API_KEY"

# Deploy
echo "Deploying..."
railway up

echo "✅ Backend deployment initiated!"
echo "Remember to:"
echo "1. Set environment variables in Railway dashboard"
echo "2. Note your Railway URL for frontend configuration"
