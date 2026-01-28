
import asyncio
import os
import sys

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()

from forecasting_tools import MetaculusApi

def check_api_capabilities():
    print("Checking MetaculusApi capabilities...")
    
    # Check if we can fetch questions with specific statuses
    # The user suggested fetching 'upcoming' and 'open'
    
    # Note: MetaculusApi methods might be different. Let's inspect the class or try headers.
    # We will try to list questions for the tournament.
    
    print(f"Tournament ID: {tournament_id}")
    
    # Inspect valid attributes/methods
    print("\nMetaculusApi attributes:")
    for attr in dir(MetaculusApi):
        if not attr.startswith("_"):
            print(f"- {attr}")
            
    # Try to find a method that looks like 'get_questions'
    # And check its signature using inspect
    import inspect
    
    if hasattr(MetaculusApi, 'get_questions'):
        print("\nSignature of get_questions:")
        print(inspect.signature(MetaculusApi.get_questions))
        
    print("\nFinding site-packages location for forecasting_tools...")
    import forecasting_tools
    print(os.path.dirname(forecasting_tools.__file__))

if __name__ == "__main__":
    check_api_capabilities()
