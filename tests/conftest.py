import sys
from pathlib import Path

# Add the project root (1-mlops-kickoff-repo) to PYTHONPATH for tests
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))