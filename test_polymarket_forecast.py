"""
Test forecast on any question (Polymarket, Metaculus, or free-form text).

This script runs the ensemble forecasting agent on a given question 
and saves output to the forecastingoutput folder.
"""

import asyncio
import os
import sys
from datetime import datetime

# Add parent directory to path for imports
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

try:
    import dotenv
    dotenv.load_dotenv()
except ImportError:
    from dotenv import load_dotenv
    load_dotenv()

from bot.agent.agent_experiment import run_ensemble_forecast


async def main():
    # Polymarket question about Kristi Noem
    question = """Will Kristi Noem be out as DHS Secretary by March 31, 2026?

Context: This is from a Polymarket prediction market. The question resolves YES if Kristi Noem 
is no longer serving as the U.S. Secretary of Homeland Security by March 31, 2026, whether 
through resignation, firing, or any other reason. It resolves NO if she is still serving in 
that position on March 31, 2026.

Current date: January 28, 2026.
Resolution date: March 31, 2026."""

    print("=" * 60)
    print("RUNNING TEST FORECAST")
    print("=" * 60)
    print(f"Question: {question[:100]}...")
    print()

    # Run the ensemble forecast (not publishing to Metaculus since this is Polymarket)
    result = await run_ensemble_forecast(
        question=question,
        publish_to_metaculus=False,  # Don't publish - this is a test
        community_prior=None,
        use_react=True,
    )

    # Save result summary to forecastingoutput
    output_dir = "forecastingoutput"
    os.makedirs(output_dir, exist_ok=True)
    
    timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
    summary_file = f"{output_dir}/test_polymarket_{timestamp}.md"
    
    with open(summary_file, "w", encoding="utf-8") as f:
        f.write(f"# Test Forecast: Kristi Noem / DHS Secretary\n\n")
        f.write(f"**Timestamp:** {timestamp}\n\n")
        f.write(f"**Final Probability:** {result['final_probability']:.1%}\n\n")
        f.write(f"## Summary\n\n{result['summary_text']}\n\n")
        f.write(f"## Full Reasoning\n\n{result.get('full_reasoning', 'N/A')}\n")
    
    print()
    print("=" * 60)
    print(f"FINAL PREDICTION: {result['final_probability']:.1%}")
    print("=" * 60)
    print(f"\nOutput saved to: {summary_file}")
    print(f"Detailed log: {output_dir}/forecast_*.txt")
    
    return result


if __name__ == "__main__":
    asyncio.run(main())
