"""
Root-level app.py for HuggingFace Spaces deployment.
Imports and exports the FastAPI app from server.app
"""

from server.app import app

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)
