import os
import sys

# Add project root directory to sys.path so backend and model packages resolve
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from backend.app import app

# Vercel serverless entrypoint
# The Flask WSGI instance 'app' is loaded directly
