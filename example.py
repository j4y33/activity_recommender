#!/usr/bin/env python3
"""
Example usage of the Activity Recommendation Pipeline

Simple script to test the pipeline with different user requests.
"""

import asyncio
from pipeline import get_activity_recommendations, run_interactive_conversation


async def test_basic_requests():
    """Test basic activity recommendation requests."""
    
    print("🎯 TESTING BASIC ACTIVITY RECOMMENDATIONS")
    print("=" * 50)
    
    test_cases = [
        "I want to go hiking near Prague this weekend",
        "Looking for running routes in Vienna city center, about 30 minutes", 
        "Any good cycling paths around San Francisco for a beginner?",
        "Indoor rock climbing near downtown Boston"
    ]
    
    for i, request in enumerate(test_cases, 1):
        print(f"\n📝 TEST {i}: {request}")
        print("-" * 40)
        
        try:
            response = await get_activity_recommendations(request)
            
            print(f"💬 Assistant: {response.conversation_message}")
            
            if response.recommendations:
                print(f"\n📋 Found {len(response.recommendations)} recommendations:")
                for j, rec in enumerate(response.recommendations, 1):
                    print(f"  {j}. {rec.activity_name}")
                    print(f"     📍 {rec.location}")
                    print(f"     ⏱️ {rec.duration_estimate} • 🎯 {rec.difficulty_level}")
            else:
                print("No recommendations found.")
                
        except Exception as e:
            print(f"❌ Error: {e}")
    
    print(f"\n✅ Basic tests completed!")


async def demo_conversation_flow():
    """Demonstrate the interactive conversation flow."""
    
    print("\n🎭 DEMONSTRATION: Interactive Conversation")
    print("=" * 50)
    
    await run_interactive_conversation(
        "I want to go running in Prague, starting from the city center"
    )


async def main():
    """Run all examples."""
    
    print("🚀 ACTIVITY RECOMMENDATION PIPELINE - EXAMPLES")
    print("=" * 60)
    print("Testing our type-safe pipeline with Agents SDK integration")
    print()
    
    # Test basic requests
    # await test_basic_requests()
    
    # Demo interactive conversation
    await demo_conversation_flow()
    
    print("\n🎉 All examples completed successfully!")
    print("\nNext steps:")
    print("- Add your API keys to .env file")
    print("- Run: python example.py")
    print("- Or use individual functions in your own code")


if __name__ == "__main__":
    asyncio.run(main()) 