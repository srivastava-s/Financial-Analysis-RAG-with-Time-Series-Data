# Vercel Deployment Guide

This guide will help you deploy the Financial Analysis RAG System to Vercel.

## 🚀 Quick Deployment

### Option 1: Deploy via Vercel Dashboard

1. **Go to [Vercel Dashboard](https://vercel.com/dashboard)**
2. **Click "New Project"**
3. **Import your GitHub repository**: `srivastava-s/Financial-Analysis-RAG-with-Time-Series-Data`
4. **Configure the project**:
   - Framework Preset: `Other`
   - Root Directory: `./` (leave as default)
   - Build Command: Leave empty (Vercel will auto-detect)
   - Output Directory: Leave empty
5. **Click "Deploy"**

### Option 2: Deploy via Vercel CLI

1. **Install Vercel CLI**:
   ```bash
   npm i -g vercel
   ```

2. **Login to Vercel**:
   ```bash
   vercel login
   ```

3. **Deploy from your project directory**:
   ```bash
   vercel
   ```

## 📁 Project Structure for Vercel

The project has been optimized for Vercel deployment with the following key files:

```
├── index.py                 # Main entry point for Vercel
├── src/app_vercel.py        # Vercel-optimized Flask app
├── requirements.txt         # Minimal dependencies
├── vercel.json             # Vercel configuration
├── runtime.txt             # Python version specification
└── .vercelignore           # Files to exclude from deployment
```

## ⚙️ Configuration Files

### vercel.json
```json
{
  "version": 2,
  "builds": [
    {
      "src": "index.py",
      "use": "@vercel/python"
    }
  ],
  "routes": [
    {
      "src": "/(.*)",
      "dest": "index.py"
    }
  ],
  "functions": {
    "index.py": {
      "runtime": "python3.9"
    }
  }
}
```

### requirements.txt
```
Flask==3.1.1
Werkzeug==3.1.3
```

### runtime.txt
```
python-3.9
```

## 🔧 Environment Variables (Optional)

If you want to add real API keys later, you can set them in Vercel:

1. Go to your project settings in Vercel
2. Navigate to "Environment Variables"
3. Add the following variables:
   - `OPENAI_API_KEY`
   - `NEWS_API_KEY`
   - `YAHOO_FINANCE_API_KEY`
   - `FRED_API_KEY`

## 🌐 Accessing Your App

After deployment, Vercel will provide you with:
- **Production URL**: `https://your-project-name.vercel.app`
- **Preview URLs**: For each deployment

## 🐛 Troubleshooting

### Common Issues:

1. **404 Error**: 
   - Ensure `index.py` exists in the root directory
   - Check that `vercel.json` points to the correct file

2. **Import Errors**:
   - Make sure all dependencies are in `requirements.txt`
   - Check that the Python version in `runtime.txt` is supported

3. **Build Failures**:
   - Check the build logs in Vercel dashboard
   - Ensure all files are properly committed to GitHub

### Debug Steps:

1. **Check Vercel Build Logs**:
   - Go to your project in Vercel dashboard
   - Click on the latest deployment
   - Check the build logs for errors

2. **Test Locally**:
   ```bash
   python index.py
   ```

3. **Verify Dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

## 📱 Features Available

The deployed app includes:
- ✅ Financial analysis interface
- ✅ Stock symbol analysis (AAPL, MSFT, GOOGL, TSLA, AMZN)
- ✅ Risk assessment
- ✅ Investment insights
- ✅ Sample data generation
- ✅ Responsive web design

## 🔄 Updates

To update your deployment:
1. Make changes to your code
2. Commit and push to GitHub
3. Vercel will automatically redeploy

## 📞 Support

If you encounter issues:
1. Check the Vercel documentation
2. Review the build logs
3. Ensure all configuration files are correct
4. Verify the Python version compatibility

---

**Note**: The current deployment uses sample data for demonstration. For production use, you'll need to integrate real financial APIs and add proper authentication.
