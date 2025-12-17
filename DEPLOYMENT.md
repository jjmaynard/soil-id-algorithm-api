# Deployment Guide: Soil ID Algorithm API

## Architecture Overview

The API provides three endpoints to handle the `SoilListOutputData` challenge:

### 1. **Separate Endpoints** (Recommended for flexibility)
- `POST /api/list-soils`: Get soil components at a location
- `POST /api/rank-soils`: Rank soils with field measurements

**Data Flow:**
```
Client → list-soils → Response (SoilListOutputData as JSON)
        ↓ (store client-side)
Client → rank-soils (with stored data + field measurements) → Response
```

### 2. **Combined Endpoint** (Recommended for simplicity)
- `POST /api/analyze-soil`: Single request that does both operations

**Data Flow:**
```
Client → analyze-soil (location + field measurements) → Response
```

## Vercel Deployment

### Prerequisites
1. Vercel account
2. Vercel CLI: `npm install -g vercel`
3. GitHub repository (optional but recommended)

### Deployment Steps

#### Option 1: Deploy via Vercel CLI

```bash
# 1. Install Vercel CLI
npm install -g vercel

# 2. Login to Vercel
vercel login

# 3. Deploy from project root
vercel

# 4. Follow prompts to configure project
# - Set up and deploy? Yes
# - Which scope? Select your account
# - Link to existing project? No
# - Project name? soil-id-algorithm-api
# - Directory? ./
# - Auto-detected settings? Yes
```

#### Option 2: Deploy via GitHub Integration

1. Push code to GitHub
2. Go to [vercel.com](https://vercel.com)
3. Click "Import Project"
4. Select your GitHub repository
5. Configure:
   - Framework Preset: Other
   - Root Directory: ./
   - Build Command: (leave empty)
   - Output Directory: (leave empty)
6. Click "Deploy"

### Configuration Files

The project includes:

**`vercel.json`**: Vercel configuration
```json
{
  "version": 2,
  "builds": [{"src": "api/main.py", "use": "@vercel/python"}],
  "routes": [{"src": "/(.*)", "dest": "api/main.py"}]
}
```

**`requirements-api.txt`**: API-specific dependencies
- Includes FastAPI, Uvicorn, and all base requirements

### Environment Variables

If you need to add environment variables:

```bash
# Via CLI
vercel env add VARIABLE_NAME

# Or via Vercel Dashboard
# Project Settings → Environment Variables
```

### Testing Deployment

Once deployed, test with:

```bash
# Test health endpoint
curl https://your-project.vercel.app/

# Test list-soils
curl -X POST https://your-project.vercel.app/api/list-soils \
  -H "Content-Type: application/json" \
  -d '{"lon": -101.9733687, "lat": 33.81246789}'

# Test combined endpoint
curl -X POST https://your-project.vercel.app/api/analyze-soil \
  -H "Content-Type: application/json" \
  -d '{
    "lon": -101.9733687,
    "lat": 33.81246789,
    "soilHorizon": ["Sandy loam"],
    "topDepth": [0],
    "bottomDepth": [20],
    "rfvDepth": ["0-1%"],
    "lab_Color": [[50.5, 5.2, 20.1]],
    "pSlope": 5.0
  }'
```

## Local Development

### Setup

```bash
# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements-api.txt

# Run development server
uvicorn api.main:app --reload --port 8000
```

### Test Locally

```bash
# Open browser to view interactive docs
open http://localhost:8000/docs

# Or use curl
curl -X POST http://localhost:8000/api/list-soils \
  -H "Content-Type: application/json" \
  -d '{"lon": -101.9733687, "lat": 33.81246789}'
```

## API Usage Examples

### Approach 1: Separate Calls (Client stores intermediate data)

```javascript
// Step 1: Get soil list
const listResponse = await fetch('https://your-api.vercel.app/api/list-soils', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ lon: -101.97, lat: 33.81 })
});
const soilListData = await listResponse.json();

// Step 2: Store data client-side (in state, localStorage, etc.)
localStorage.setItem('soilListData', JSON.stringify(soilListData));

// Step 3: Later, rank with field measurements
const rankResponse = await fetch('https://your-api.vercel.app/api/rank-soils', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    lon: -101.97,
    lat: 33.81,
    ...soilListData,  // Spread the stored data
    soilHorizon: ['Sandy loam'],
    topDepth: [0],
    bottomDepth: [20],
    // ... other field measurements
  })
});
const rankResult = await rankResponse.json();
```

### Approach 2: Combined Call (Simpler)

```javascript
// Single call with all data
const response = await fetch('https://your-api.vercel.app/api/analyze-soil', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({
    lon: -101.97,
    lat: 33.81,
    soilHorizon: ['Sandy loam'],
    topDepth: [0],
    bottomDepth: [20],
    rfvDepth: ['0-1%'],
    lab_Color: [[50.5, 5.2, 20.1]],
    pSlope: 5.0,
    pElev: 800.0,
    bedrock: null,
    cracks: false
  })
});
const result = await response.json();
```

## Performance Considerations

### Vercel Serverless Limits
- **Timeout**: 10 seconds (Hobby), 60 seconds (Pro)
- **Memory**: 1024 MB (Hobby), 3008 MB (Pro)
- **Package Size**: 50 MB max

### Optimization Tips
1. **Use Combined Endpoint** when always calling both functions
2. **Add Caching** for frequently accessed locations (requires external service)
3. **Upgrade to Pro** if functions exceed 10-second timeout
4. **Monitor Cold Starts** and optimize dependencies if needed

## Troubleshooting

### Common Issues

**"Module not found" errors:**
- Check `PYTHONPATH` in `vercel.json`
- Ensure all dependencies are in `requirements-api.txt`

**Timeout errors:**
- Functions exceeding 10s need Pro plan
- Consider optimizing database queries
- Add timeout handling in client code

**Large payload errors:**
- The `SoilListOutputData` can be large (CSV strings)
- Consider compressing or paginating if needed
- Monitor request/response sizes

## Next Steps

1. Deploy to Vercel
2. Test all endpoints
3. Update client application to use API
4. Add authentication if needed (API keys, OAuth)
5. Set up monitoring (Vercel Analytics, Sentry)
6. Configure custom domain

## Support

- Vercel Docs: https://vercel.com/docs
- FastAPI Docs: https://fastapi.tiangolo.com
