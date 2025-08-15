"""
Vercel entry point for Financial Analysis RAG System.
This file serves as the main entry point for Vercel deployment.
"""

from src.app_vercel import app

# Vercel requires this to be named 'app'
app.debug = False
