#  Unified Model for Smart Mobility Simulation
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows: venv\Scripts\activate

# run backend unicorn server  on http://127.0.0.1:8000/docs
uvicorn api.main:app --reload