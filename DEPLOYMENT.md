# Deployment Environment Variables

## For Frontend (Vercel)
Set these in your Vercel dashboard:

```
NEXT_PUBLIC_API_URL=https://your-backend-url.railway.app
```

## For Backend (Railway/Render)
Set these in your backend hosting service:

```
SUPABASE_URL=your_supabase_url
SUPABASE_KEY=your_supabase_anon_key
GEMINI_API_KEY=your_gemini_api_key
```

## Deployment URLs
- Frontend: https://your-app.vercel.app
- Backend: https://your-backend-url.railway.app  
- Database: Already hosted on Supabase

## Quick Deploy Commands

### Frontend to Vercel:
```bash
npx vercel --prod
```

### Backend to Railway:
```bash
# Install Railway CLI
npm install -g @railway/cli

# Login and deploy
railway login
railway new
railway up
```

## Post-Deployment Checklist
- [ ] Update NEXT_PUBLIC_API_URL in Vercel
- [ ] Set backend environment variables
- [ ] Test all endpoints
- [ ] Update CORS settings if needed
```
