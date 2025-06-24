"""
Activity Recommendation Pipeline

Orchestrates the type-safe agents and tools for activity recommendations:
1. intent_agent: Parses user request + gets weather
2. extraction_agent: Searches web + extracts activities  
3. conversation_agent: Creates conversational responses + handles feedback

Can be run programmatically or with --interactive flag for chat interface.
"""

import asyncio
import argparse
import sys
from typing import List, Optional
from agents import Runner

from core.config import initialize_config, get_instructor_client, get_config
from core.agents import intent_agent, extraction_agent, conversation_agent
from core.models import UserIntent, ExtractedActivity, ConversationalResponse

# Initialize configuration to ensure API keys are loaded
config = initialize_config()


async def gather_preferences(user_request: str) -> str:
    """
    Analyze user request and gather additional preferences to create a personalized search query.
    
    Args:
        user_request: Initial user request
        
    Returns:
        Enhanced request with gathered preferences
    """
    print("🧠 Analyzing request for personalization...")
    
    try:
        client = get_instructor_client()
        config = get_config()
        
        # Use LLM to determine if we need more information
        analysis_prompt = f"""
        Analyze this activity request and determine if we need more information to provide great recommendations.
        
        USER REQUEST: "{user_request}"
        
        Consider:
        1. Is the activity type clear? (hiking, running, cycling, etc.)
        2. Is the location specific enough?
        3. Are preferences mentioned? (difficulty, duration, equipment, indoor/outdoor)
        4. Is the weather consideration clear?
        
        If the request is specific enough, return "SUFFICIENT"
        If we need more information, return specific questions to ask, separated by "|"
        
        Examples:
        - "I want exercise" → "What type of exercise do you prefer?|How long do you want to exercise?|Indoor or outdoor?"
        - "hiking in Prague" → "SUFFICIENT"
        - "something fun" → "What type of activity interests you?|Do you prefer indoor or outdoor?|How much time do you have?"
        """
        
        from openai import AsyncOpenAI
        openai_client = AsyncOpenAI(api_key=config.api.openai_api_key)
        
        response = await openai_client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at understanding activity requests and determining when more information is needed."},
                {"role": "user", "content": analysis_prompt}
            ],
            temperature=0.1,
            max_tokens=200
        )
        
        analysis_result = response.choices[0].message.content.strip()
        
        if analysis_result == "SUFFICIENT":
            print("✅ Request is specific enough")
            return user_request
        
        # Need to gather more preferences
        print("🤔 Request needs more details, but proceeding with current info")
        print(f"💡 Suggestions for improvement: {analysis_result}")
        
        # For now, just proceed with the original request
        # In a full implementation, you could ask these questions interactively
        return user_request
        
    except Exception as e:
        print(f"⚠️ Preference analysis error: {e}, proceeding with original request")
        return user_request


def enhance_search_query(user_intent: UserIntent) -> str:
    """
    Enhance the search query with personal touches and geographic specificity.
    
    Args:
        user_intent: Parsed user intent
        
    Returns:
        Enhanced search query
    """
    
    base_query = user_intent.search_query
    location = user_intent.location
    
    # Add geographic specificity to avoid confusion with US cities
    location_mappings = {
        "vienna": "Vienna Austria",
        "prague": "Prague Czech Republic", 
        "budapest": "Budapest Hungary",
        "berlin": "Berlin Germany",
        "munich": "Munich Germany",
        "zurich": "Zurich Switzerland",
        "amsterdam": "Amsterdam Netherlands",
        "stockholm": "Stockholm Sweden",
        "copenhagen": "Copenhagen Denmark",
        "oslo": "Oslo Norway",
        "helsinki": "Helsinki Finland"
    }
    
    # Check if we need to add country specification
    location_lower = location.lower()
    for city, full_location in location_mappings.items():
        if city in location_lower and "austria" not in location_lower and "czech" not in location_lower:
            # Replace the location in the query with the full geographic name
            base_query = base_query.replace(location, full_location)
            break
    
    # Add weather-specific enhancements
    weather = user_intent.weather_condition.lower()
    enhancements = []
    
    if "sunny" in weather or "clear" in weather:
        enhancements.append("sunny weather")
    elif "rain" in weather:
        if user_intent.indoor_outdoor == "outdoor":
            enhancements.append("covered routes")
    elif "cloudy" in weather or "overcast" in weather:
        enhancements.append("day trip")
    
    # Add time-based enhancements
    import datetime
    current_hour = datetime.datetime.now().hour
    
    if 6 <= current_hour <= 10:
        enhancements.append("morning")
    elif 17 <= current_hour <= 20:
        enhancements.append("evening")
    elif current_hour >= 21 or current_hour <= 5:
        if user_intent.activity_type in ["running", "cycling"]:
            enhancements.append("well-lit safe routes")
    
    # Combine base query with enhancements
    if enhancements:
        enhanced_query = f"{base_query} {' '.join(enhancements[:2])}"  # Max 2 enhancements
    else:
        enhanced_query = base_query
    
    print(f"🎯 Enhanced search query: '{enhanced_query}'")
    return enhanced_query


async def get_activity_recommendations(user_request: str, bypass_clarification: bool = False) -> ConversationalResponse:
    """
    Get activity recommendations using the orchestrated agent pipeline.
    
    Args:
        user_request: User's natural language request
        bypass_clarification: Whether to bypass clarification for proceed responses
        
    Returns:
        ConversationalResponse with recommendations and conversational message
    """
    
    print(f"🚀 Starting activity recommendation pipeline")
    print(f"📝 User request: {user_request}")
    
    try:
        # Step 0: Gather preferences and personalize request
        enhanced_request = await gather_preferences(user_request)
        
        # Step 1: Parse user intent and get weather data
        print("\n🧠 Step 1: Parsing user intent...")
        intent_result = await Runner.run(intent_agent, enhanced_request)
        user_intent: UserIntent = intent_result.final_output
        
        print(f"✅ Intent parsed:")
        print(f"   Activity: {user_intent.activity_type}")
        print(f"   Location: {user_intent.location}")
        print(f"   Weather: {user_intent.weather_condition}")
        print(f"   Search query: {user_intent.search_query}")
        print(f"   Generic query: {user_intent.is_generic}")
        print(f"   Needs clarification: {user_intent.needs_clarification}")
        
        # Check if query needs clarification
        if user_intent.needs_clarification and not bypass_clarification:
            print("🔍 Query is very generic - offering clarification...")
            clarification_message = _generate_clarification_message(user_intent)
            return ConversationalResponse(
                recommendations=[],
                conversation_message=clarification_message
            )
        
        # Enhance search query with personal touches
        enhanced_search_query = enhance_search_query(user_intent)
        user_intent.search_query = enhanced_search_query
        
        # Step 2: Extract activities from web search
        print(f"\n🕷️ Step 2: Extracting activities...")
        extraction_input = f"""
        UserIntent: {user_intent.model_dump_json(indent=2)}
        
        Use this UserIntent to search for and extract relevant activities.
        Focus on finding SPECIFIC activities, not generic categories.
        Extract detailed metrics when available from sources.
        """
        
        extraction_result = await Runner.run(extraction_agent, extraction_input)
        extracted_activities: List[ExtractedActivity] = extraction_result.final_output
        
        print(f"✅ Extracted {len(extracted_activities)} activities")
        for activity in extracted_activities:
            details_note = " (with details)" if activity.details_available else ""
            print(f"   - {activity.activity_name} (relevance: {activity.relevance_score:.2f}){details_note}")
        
        # Step 3: Create conversational response
        print(f"\n💬 Step 3: Generating conversational response...")
        conversation_input = f"""
        Create initial recommendations from these extracted activities:
        
        User Request: {user_request}
        Extracted Activities: {[activity.model_dump() for activity in extracted_activities]}
        
        Convert ExtractedActivity objects to ActivityRecommendation objects and create a friendly response.
        Include detailed information when available from sources.
        This is the first turn, so don't use analyze_feedback tool.
        """
        
        conversation_result = await Runner.run(conversation_agent, conversation_input)
        response: ConversationalResponse = conversation_result.final_output
        
        print(f"✅ Generated response with {len(response.recommendations)} recommendations")
        print(f"💬 Message: {response.conversation_message}")
        
        return response
        
    except Exception as e:
        print(f"❌ Pipeline error: {e}")
        # Return fallback response
        return ConversationalResponse(
            recommendations=[],
            conversation_message=f"I encountered an error processing your request: {e}. Please try again with a different request."
        )


def _generate_clarification_message(user_intent: UserIntent) -> str:
    """
    Generate a clarification message for generic queries.
    
    Args:
        user_intent: Parsed intent with generic flag
        
    Returns:
        Clarification message with specific questions
    """
    
    activity = user_intent.activity_type
    location = user_intent.location
    
    # Generate relevant clarification questions based on activity type
    questions = []
    
    if activity in ["running", "jogging"]:
        questions = [
            "What distance are you thinking? (e.g., 5km, 10km, or just time-based like 30 minutes)",
            "Do you prefer challenging routes with hills or flatter terrain?", 
            "Any preference for trails vs. paved paths?",
            "Do you have a specific starting point in mind?"
        ]
    elif activity in ["hiking", "walking"]:
        questions = [
            "What difficulty level? (easy, moderate, or challenging)",
            "How much time do you want to spend? (30 minutes, 1-2 hours, half day)",
            "Do you prefer flat walks or some elevation gain?",
            "Any specific area or starting point you'd like?"
        ]
    elif activity in ["cycling", "biking"]:
        questions = [
            "What distance are you thinking? (short ride, 10-20km, longer tour)",
            "Do you prefer bike paths/lanes or are you comfortable with mixed traffic?",
            "Easy flat route or okay with some hills?",
            "Recreation cycling or more sporty/fitness focused?"
        ]
    else:
        # Generic questions for other activities
        questions = [
            f"What level of {activity} are you looking for? (beginner, intermediate, advanced)",
            "How much time do you want to spend?",
            "Any specific preferences for location or type?",
            "Indoor, outdoor, or either is fine?"
        ]
    
    # Select 2-3 most relevant questions
    selected_questions = questions[:3]
    questions_text = "\n".join(f"• {q}" for q in selected_questions)
    
    return f"""I found your request for {activity} in {location}! To give you the best recommendations, could you help me with a few details?

{questions_text}

Or if you'd like, I can proceed with general {activity} options in {location} and you can refine from there. Just say "proceed" or "go ahead"!"""


async def handle_user_feedback(
    user_feedback: str, 
    original_request: str,
    previous_response: ConversationalResponse
) -> ConversationalResponse:
    """
    Handle user feedback - can trigger new search or refinement based on feedback analysis.
    
    Args:
        user_feedback: User's feedback on previous recommendations
        original_request: Original user request for context
        previous_response: Previous recommendations for context
        
    Returns:
        Updated ConversationalResponse based on feedback
    """
    
    print(f"💭 Processing user feedback: '{user_feedback}'")
    
    try:
        # First, analyze the feedback to understand intent
        feedback_input = f"""
        Analyze this user feedback to determine what action to take.
        
        Original Request: {original_request}
        User Feedback: {user_feedback}
        Previous Recommendations: {[rec.model_dump() for rec in previous_response.recommendations]}
        
        Use analyze_feedback tool to classify the feedback.
        """
        
        feedback_result = await Runner.run(conversation_agent, feedback_input)
        feedback_analysis = feedback_result.final_output
        
        # Debug: Print the result structure
        print(f"🔍 Debug - feedback result type: {type(feedback_result)}")
        print(f"🔍 Debug - has messages: {hasattr(feedback_result, 'messages')}")
        
        # Look for analyze_feedback tool usage by examining the conversation
        feedback_status = "refinement"  # default
        
        # Check for key phrases that indicate new search intent
        if hasattr(feedback_analysis, 'conversation_message'):
            message = feedback_analysis.conversation_message.lower()
            user_feedback_lower = user_feedback.lower()
            
            # Handle "proceed" responses to clarification questions
            if any(word in user_feedback_lower for word in ["proceed", "go ahead", "continue", "yes"]):
                print("🎯 User wants to proceed with general search")
                # Use original request but force generic processing
                new_response = await get_activity_recommendations(original_request, bypass_clarification=True)
                return new_response
            
            # Direct detection from user feedback
            new_search_indicators = [
                "rather", "instead", "prefer", "different", "something else",
                "how about", "what about", "completely different"
            ]
            
            # More specific refinement indicators
            refinement_indicators = [
                "longer", "shorter", "easier", "harder", "closer", "further",
                "outskirts", "more time", "less time", "duration", "distance"
            ]
            
            if any(indicator in user_feedback_lower for indicator in new_search_indicators):
                feedback_status = "new_search"
                print(f"🎯 Detected new search from user feedback keywords")
            elif any(indicator in user_feedback_lower for indicator in refinement_indicators):
                feedback_status = "refinement"
                print(f"🎯 Detected refinement from user feedback keywords")
            elif "satisfied" in message or "perfect" in message or "happy" in message:
                feedback_status = "satisfied"
            elif "new_search" in message:
                feedback_status = "new_search"
        
        print(f"🎯 Final detected feedback type: {feedback_status}")
        
        # If user wants new search, create new request and run full pipeline
        if feedback_status == "new_search":
            print("🔄 User wants different activities - starting new search...")
            
            # Create new search request based on user feedback (ignore original request)
            new_request = user_feedback
            
            # Run complete new search
            new_response = await get_activity_recommendations(new_request)
            
            print(f"✅ New search completed with {len(new_response.recommendations)} recommendations")
            return new_response
        
        # For refinement, intelligently merge feedback with original request
        elif feedback_status == "refinement":
            print("🔄 User wants refinement - updating preferences...")
            
            # Extract key context from original request
            refined_request = await _merge_feedback_with_original(original_request, user_feedback)
            
            # Run new search with refined request
            refined_response = await get_activity_recommendations(refined_request)
            
            print(f"✅ Refinement completed with {len(refined_response.recommendations)} recommendations")
            return refined_response
        
        # For satisfied, use the existing response
        print(f"✅ Feedback processed as {feedback_status}")
        print(f"💬 Response: {feedback_analysis.conversation_message}")
        
        return feedback_analysis
        
    except Exception as e:
        print(f"❌ Feedback processing error: {e}")
        return ConversationalResponse(
            recommendations=previous_response.recommendations,
            conversation_message="I had trouble understanding your feedback. Could you please try rephrasing what you'd like to change?"
        )


async def _merge_feedback_with_original(original_request: str, user_feedback: str) -> str:
    """
    Intelligently merge user feedback with original request to preserve context.
    
    Args:
        original_request: Original user request with activity type, location, etc.
        user_feedback: User's refinement feedback
        
    Returns:
        Merged request that preserves key context while incorporating feedback
    """
    
    # Use LLM to intelligently merge the requests
    from core.config import get_openai_client
    
    try:
        client = get_openai_client()
        
        prompt = f"""
        You need to merge a user's original request with their refinement feedback to create a new search request.
        
        PRESERVE THESE FROM ORIGINAL:
        - Activity type (running, hiking, cycling, etc.)
        - Location (city, area, country)
        - Any specific preferences that weren't contradicted
        
        UPDATE BASED ON FEEDBACK:
        - Duration/distance preferences
        - Difficulty level
        - Specific location refinements (outskirts, city center, etc.)
        - Equipment or terrain preferences
        
        ORIGINAL REQUEST: "{original_request}"
        USER FEEDBACK: "{user_feedback}"
        
        Examples:
        Original: "running route in Vienna, hard"
        Feedback: "longer routes, could also be more in the outskirts"
        Result: "long running routes in Vienna outskirts, hard difficulty"
        
        Original: "hiking near San Francisco"  
        Feedback: "something easier and closer to downtown"
        Result: "easy hiking trails close to downtown San Francisco"
        
        Create a new search request that intelligently combines both:
        """
        
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": "You are an expert at merging user requests while preserving key context."},
                {"role": "user", "content": prompt}
            ],
            temperature=0.1,
            max_tokens=100
        )
        
        merged_request = response.choices[0].message.content.strip()
        print(f"🔀 Merged request: '{merged_request}'")
        return merged_request
        
    except Exception as e:
        print(f"⚠️ Request merging failed, using feedback only: {e}")
        return user_feedback


async def run_interactive_conversation(user_request: str, max_turns: int = 3) -> None:
    """
    Run an interactive conversation with feedback loops.
    
    Args:
        user_request: Initial user request
        max_turns: Maximum conversation turns
    """
    
    print("🎭 INTERACTIVE ACTIVITY RECOMMENDATION CONVERSATION")
    print("=" * 60)
    
    # Get initial recommendations
    response = await get_activity_recommendations(user_request)
    
    # Display initial response
    print(f"\n🤖 ASSISTANT:")
    print(f"💬 {response.conversation_message}")
    if response.recommendations:
        print(f"\n📋 RECOMMENDATIONS:")
        for i, rec in enumerate(response.recommendations, 1):
            print(f"{i}. {rec.activity_name}")
            print(f"   📍 {rec.location}")
            print(f"   📝 {rec.description}")
            print(f"   ⏱️ {rec.duration_estimate} • 🎯 {rec.difficulty_level}")
            print()
    
    # Simulate feedback turns (in real app, this would be user input)
    sample_feedback = [
        "These look too challenging for me, do you have easier options?",
        "Perfect! These look great, thank you!"
    ]
    
    current_response = response
    
    for turn in range(min(len(sample_feedback), max_turns - 1)):
        print(f"\n{'='*20} TURN {turn + 2} {'='*20}")
        
        feedback = sample_feedback[turn]
        print(f"👤 USER FEEDBACK: {feedback}")
        
        # Process feedback
        feedback_response = await handle_user_feedback(
            feedback, user_request, current_response
        )
        
        print(f"\n🤖 ASSISTANT:")
        print(f"💬 {feedback_response.conversation_message}")
        
        if feedback_response.recommendations:
            print(f"\n📋 UPDATED RECOMMENDATIONS:")
            for i, rec in enumerate(feedback_response.recommendations, 1):
                print(f"{i}. {rec.activity_name} - {rec.location}")
        
        current_response = feedback_response
        
        # Check if conversation should end
        if "perfect" in feedback.lower() or "great" in feedback.lower():
            print(f"\n✅ User satisfied - conversation complete!")
            break
    
    print(f"\n🎯 Conversation completed successfully!")


# Example usage functions

async def quick_demo():
    """Quick demo of the pipeline."""
    print("🎯 QUICK PIPELINE DEMO")
    print("=" * 30)
    
    test_requests = [
        "I want to go hiking near Prague this weekend",
        "Looking for running routes in Vienna city center, about 30 minutes"
    ]
    
    for request in test_requests:
        print(f"\n📝 Testing: {request}")
        response = await get_activity_recommendations(request)
        print(f"💬 Response: {response.conversation_message}")
        print(f"📊 Found {len(response.recommendations)} recommendations")


async def main():
    """Main entry point with command-line argument parsing."""
    parser = argparse.ArgumentParser(
        description="Activity Recommendation Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python pipeline.py --interactive              # Launch chat interface
  python pipeline.py --demo                     # Run quick demo
  python pipeline.py --test "hiking in Prague"  # Test single request
        """.strip()
    )
    
    group = parser.add_mutually_exclusive_group()
    group.add_argument(
        "--interactive", 
        action="store_true",
        help="Launch interactive chat interface"
    )
    group.add_argument(
        "--demo",
        action="store_true", 
        help="Run quick demo with sample requests"
    )
    group.add_argument(
        "--test",
        type=str,
        metavar="REQUEST",
        help="Test a single activity request"
    )
    
    args = parser.parse_args()
    
    try:
        # Initialize configuration
        config = initialize_config()
        
        # Validate setup
        validation = config.validate_setup()
        missing_keys = [key for key, status in validation.items() if not status]
        
        if missing_keys:
            print("❌ CONFIGURATION ERROR")
            print("Missing or invalid API keys:")
            for key in missing_keys:
                print(f"  - {key}")
            print("\nPlease add the required API keys to your .env file:")
            print("  OPENAI_API_KEY=your_openai_key")
            print("  OPENWEATHER_API_KEY=your_openweather_key") 
            print("  FIRECRAWL_API_KEY=your_firecrawl_key")
            return 1
        
        if args.interactive:
            # Launch interactive chat interface
            print("🚀 Launching interactive chat interface...")
            
            # Import ChatInterface here to avoid circular imports
            try:
                from chat import ChatInterface
                chat = ChatInterface()
                await chat.run()
            except ImportError as e:
                print(f"❌ Could not import ChatInterface: {e}")
                print("Make sure chat.py is in the same directory.")
                return 1
                
        elif args.demo:
            # Run quick demo
            await quick_demo()
            
        elif args.test:
            # Test single request
            print(f"🧪 Testing request: {args.test}")
            response = await get_activity_recommendations(args.test)
            print(f"💬 Response: {response.conversation_message}")
            print(f"📊 Found {len(response.recommendations)} recommendations")
            
            if response.recommendations:
                print("\n📋 RECOMMENDATIONS:")
                for i, rec in enumerate(response.recommendations, 1):
                    print(f"{i}. {rec.activity_name}")
                    print(f"   📍 {rec.location}")
                    print(f"   ⏱️ {rec.duration_estimate} • 🎯 {rec.difficulty_level}")
                    print()
        else:
            # No arguments provided, show help
            parser.print_help()
            print("\n💡 TIP: Use --interactive to start the chat interface!")
            return 0
            
        return 0
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye!")
        return 0
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        return 1


if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
