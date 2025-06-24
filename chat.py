#!/usr/bin/env python3
"""
Interactive Terminal Chat Interface for Activity Recommendations

A beautiful, real-time chat interface that leverages the existing agent pipeline.
Features:
- Real user input/output with proper formatting
- Conversation history and context
- Error handling and graceful failures
- Type-safe integration with existing agents
"""

import asyncio
import sys
from typing import List, Optional
from datetime import datetime

from core.config import initialize_config, get_config
from core.models import ConversationalResponse, ActivityRecommendation
from pipeline import get_activity_recommendations, handle_user_feedback


class ChatInterface:
    """Interactive terminal chat interface for activity recommendations."""
    
    def __init__(self):
        self.conversation_history: List[dict] = []
        self.current_response: Optional[ConversationalResponse] = None
        self.original_request: str = ""
        self.turn_count: int = 0
        self.max_turns: int = 5
        
    def print_header(self):
        """Print welcome header."""
        print("🏃‍♂️ ACTIVITY RECOMMENDATION ASSISTANT")
        print("=" * 50)
        print("Ask me for activity recommendations anywhere in the world!")
        print("I'll search the web, check the weather, and find perfect activities for you.")
        print()
        print("Examples:")
        print('• "I want to go hiking near Prague this weekend"')
        print('• "Looking for running routes in Vienna city center"')
        print('• "Any good cycling paths around San Francisco for beginners?"')
        print('• "Indoor rock climbing near downtown Boston"')
        print()
        print("Type 'quit', 'exit', or 'bye' to end the conversation.")
        print("=" * 50)
    
    def print_separator(self):
        """Print conversation separator."""
        print("\n" + "─" * 50 + "\n")
    
    def format_recommendations(self, recommendations: List[ActivityRecommendation]) -> None:
        """Format and display recommendations beautifully with enhanced details."""
        if not recommendations:
            print("❌ No recommendations found.")
            return
        
        print(f"📋 RECOMMENDATIONS ({len(recommendations)} found):")
        print()
        
        for i, rec in enumerate(recommendations, 1):
            # Header with activity name and location
            print(f"🎯 {i}. {rec.activity_name}")
            print(f"   📍 {rec.location}")
            
            # Description (truncated if too long)
            desc = rec.description
            if len(desc) > 120:
                desc = desc[:120] + "..."
            print(f"   📝 {desc}")
            
            # Enhanced details section - show detailed metrics when available
            detailed_metrics = []
            basic_details = []
            
            # Collect detailed metrics (these have priority)
            if hasattr(rec, 'distance') and rec.distance:
                detailed_metrics.append(f"📏 Distance: {rec.distance}")
            if hasattr(rec, 'elevation_gain') and rec.elevation_gain:
                detailed_metrics.append(f"⛰️ Elevation: {rec.elevation_gain}")
            if hasattr(rec, 'estimated_time') and rec.estimated_time:
                detailed_metrics.append(f"⏱️ Time: {rec.estimated_time}")
            if hasattr(rec, 'average_rating') and rec.average_rating:
                detailed_metrics.append(f"⭐ Rating: {rec.average_rating}")
            if hasattr(rec, 'surface_type') and rec.surface_type:
                detailed_metrics.append(f"🛤️ Surface: {rec.surface_type}")
            if hasattr(rec, 'route_type') and rec.route_type:
                detailed_metrics.append(f"🔄 Type: {rec.route_type}")
            
            # Collect basic details (fallback if no detailed metrics)
            if rec.duration_estimate and rec.duration_estimate != "varies":
                basic_details.append(f"⏱️ {rec.duration_estimate}")
            if rec.difficulty_level and rec.difficulty_level != "not specified":
                basic_details.append(f"🎯 {rec.difficulty_level}")
            if rec.indoor_outdoor:
                basic_details.append(f"🏠 {rec.indoor_outdoor}")
            
            # Display detailed metrics if available, otherwise basic details
            if detailed_metrics:
                print("   📊 Details:")
                for metric in detailed_metrics[:4]:  # Show max 4 detailed metrics
                    print(f"      {metric}")
                # Still show difficulty and indoor/outdoor if available
                extra_details = []
                if rec.difficulty_level and rec.difficulty_level != "not specified":
                    extra_details.append(f"🎯 {rec.difficulty_level}")
                if rec.indoor_outdoor:
                    extra_details.append(f"🏠 {rec.indoor_outdoor}")
                if extra_details:
                    print(f"   {' • '.join(extra_details)}")
            elif basic_details:
                print(f"   {' • '.join(basic_details)}")
            
            # Starting point if available
            if hasattr(rec, 'starting_point') and rec.starting_point:
                print(f"   📍 Start: {rec.starting_point}")
            
            # Weather recommendation (if available)
            if rec.weather_recommendation:
                print(f"   🌤️ {rec.weather_recommendation}")
            elif rec.weather_suitability and rec.weather_suitability != "any weather":
                print(f"   🌤️ {rec.weather_suitability}")
            
            # Equipment (always show, even if none needed)
            if rec.equipment_needed:
                equipment = ", ".join(rec.equipment_needed[:3])  # Show max 3 items
                if len(rec.equipment_needed) > 3:
                    equipment += f" (+ {len(rec.equipment_needed) - 3} more)"
                print(f"   🎒 Equipment: {equipment}")
            else:
                print(f"   🎒 Equipment: None needed")
            
            # Source URL (show full URL or better truncation)
            if rec.source_url:
                # Better URL display - show domain + path
                display_url = rec.source_url
                if len(display_url) > 80:
                    # Try to keep domain and meaningful path
                    from urllib.parse import urlparse
                    try:
                        parsed = urlparse(display_url)
                        domain = parsed.netloc
                        path = parsed.path
                        if len(domain + path) > 70:
                            display_url = f"{domain}{path[:50]}..."
                        else:
                            display_url = domain + path
                    except:
                        display_url = display_url[:77] + "..."
                print(f"   🔗 Source: {display_url}")
            
            print()
    
    def get_user_input(self, prompt: str = "You") -> str:
        """Get user input with nice formatting."""
        try:
            user_input = input(f"👤 {prompt}: ").strip()
            return user_input
        except (KeyboardInterrupt, EOFError):
            print("\n\n👋 Goodbye! Have a great time with your activities!")
            sys.exit(0)
    
    def should_quit(self, user_input: str) -> bool:
        """Check if user wants to quit."""
        quit_words = ['quit', 'exit', 'bye', 'goodbye', 'stop']
        return user_input.lower() in quit_words
    
    async def process_initial_request(self, user_request: str) -> bool:
        """Process the initial activity request."""
        try:
            print(f"\n🤖 Assistant: Let me find some great activities for you...")
            print("🔍 Searching the web and checking weather conditions...")
            
            # Get recommendations using existing pipeline
            self.current_response = await get_activity_recommendations(user_request)
            self.original_request = user_request
            
            # Display response
            self.print_separator()
            print(f"🤖 Assistant: {self.current_response.conversation_message}")
            print()
            self.format_recommendations(self.current_response.recommendations)
            
            # Save to history
            self.conversation_history.append({
                'type': 'user_request',
                'content': user_request,
                'timestamp': datetime.now().isoformat()
            })
            self.conversation_history.append({
                'type': 'assistant_response',
                'content': self.current_response.conversation_message,
                'recommendations': len(self.current_response.recommendations),
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print(f"\n❌ Sorry, I encountered an error: {e}")
            print("Please try again with a different request.")
            return False
    
    async def process_feedback(self, feedback: str) -> bool:
        """Process user feedback on recommendations."""
        try:
            print(f"\n🤖 Assistant: Let me understand your feedback...")
            
            # Process feedback using existing pipeline
            feedback_response = await handle_user_feedback(
                feedback, self.original_request, self.current_response
            )
            
            # Update current response
            self.current_response = feedback_response
            
            # Display response
            self.print_separator()
            print(f"🤖 Assistant: {feedback_response.conversation_message}")
            
            # Show updated recommendations if any
            if feedback_response.recommendations:
                print()
                self.format_recommendations(feedback_response.recommendations)
            
            # Save to history
            self.conversation_history.append({
                'type': 'user_feedback',
                'content': feedback,
                'timestamp': datetime.now().isoformat()
            })
            self.conversation_history.append({
                'type': 'assistant_feedback_response',
                'content': feedback_response.conversation_message,
                'recommendations': len(feedback_response.recommendations),
                'timestamp': datetime.now().isoformat()
            })
            
            return True
            
        except Exception as e:
            print(f"\n❌ Sorry, I had trouble processing your feedback: {e}")
            print("Could you please try rephrasing what you'd like to change?")
            return False
    
    def check_conversation_satisfaction(self, user_input: str) -> bool:
        """Check if user seems satisfied to end conversation."""
        satisfaction_indicators = [
            'perfect', 'great', 'excellent', 'thanks', 'thank you', 
            'that\'s all', 'looks good', 'sounds good', 'perfect!'
        ]
        return any(indicator in user_input.lower() for indicator in satisfaction_indicators)
    
    async def run(self):
        """Run the interactive chat interface."""
        self.print_header()
        
        # Get initial request
        while True:
            user_request = self.get_user_input("What activity are you looking for?")
            
            if self.should_quit(user_request):
                print("\n👋 Goodbye! Have a great time with your activities!")
                return
            
            if len(user_request.strip()) < 5:
                print("Please provide a more detailed activity request.")
                continue
            
            # Process initial request
            success = await self.process_initial_request(user_request)
            if success:
                break
        
        # Handle feedback loop
        while self.turn_count < self.max_turns:
            self.turn_count += 1
            
            self.print_separator()
            print("💭 What do you think? Would you like me to refine these recommendations?")
            print("   (Or say 'perfect' if you're happy with these results)")
            
            feedback = self.get_user_input("Your thoughts")
            
            if self.should_quit(feedback):
                print("\n👋 Goodbye! Have a great time with your activities!")
                return
            
            # Check if user is satisfied
            if self.check_conversation_satisfaction(feedback):
                print("\n✅ Wonderful! I'm glad I could help you find some great activities.")
                print("Have an amazing time exploring!")
                return
            
            # Process feedback
            await self.process_feedback(feedback)
            
            # Check turn limit
            if self.turn_count >= self.max_turns:
                print(f"\n⏰ We've reached the conversation limit.")
                print("I hope some of these recommendations are helpful!")
                return
        
        print("\n✅ Conversation complete! Enjoy your activities!")


async def main():
    """Main entry point for the chat interface."""
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
            return
        
        # Print status
        print("✅ Configuration validated successfully!")
        config.print_status()
        print()
        
        # Start chat interface
        chat = ChatInterface()
        await chat.run()
        
    except KeyboardInterrupt:
        print("\n\n👋 Goodbye! Have a great time with your activities!")
    except Exception as e:
        print(f"\n❌ Fatal error: {e}")
        print("Please check your configuration and try again.")


if __name__ == "__main__":
    asyncio.run(main()) 