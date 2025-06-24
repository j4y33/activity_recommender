"""
Function tools for the activity recommendation agents.

This module defines the capabilities that agents can use to interact with external services:
- get_weather: Real-time weather data from OpenWeather API
- search_web: Web search using Firecrawl API
- scrape_results: Extract activity data from specific URLs with full type safety
- analyze_feedback: Semantic feedback understanding for conversation flow

Each tool returns proper Pydantic models using Instructor for type safety and structured outputs.
"""

import asyncio
import json
import aiohttp
import time
from typing import List, Optional, Dict
from agents import function_tool

from core.config import get_api_key, get_config, get_instructor_client
from core.models import SearchResult, ExtractedActivity, TurnFeedback, SmartExtractionResult, PageAnalysis, ActivityCandidate

# Simple weather cache to prevent duplicate API calls
_weather_cache: Dict[str, tuple[str, float]] = {}  # location -> (weather_data, timestamp)
_cache_duration = 300  # 5 minutes


@function_tool(strict_mode=False)
async def get_weather(location: str) -> str:
    """
    Get current weather for a location using OpenWeatherMap API.
    
    Args:
        location: The city or location to get weather for (e.g., "San Francisco", "Prague")
    
    Returns:
        Weather description string (e.g., "Sunny, 18°C, light winds")
    """
    # Check cache first
    location_key = location.lower().strip()
    current_time = time.time()
    
    if location_key in _weather_cache:
        cached_data, timestamp = _weather_cache[location_key]
        if current_time - timestamp < _cache_duration:
            print(f"✅ Using cached weather data for {location}")
            return cached_data
    
    api_key = get_api_key("openweather")
    
    if not api_key:
        return "Weather data unavailable - no API key configured"
    
    try:
        url = "https://api.openweathermap.org/data/2.5/weather"
        params = {
            "q": location,
            "appid": api_key,
            "units": "metric"
        }
        
        async with aiohttp.ClientSession() as session:
            async with session.get(url, params=params) as response:
                if response.status == 200:
                    data = await response.json()
                    
                    temp = data['main']['temp']
                    description = data['weather'][0]['description']
                    wind_speed = data['wind']['speed']
                    
                    weather_result = f"{description.title()}, {temp}°C, wind {wind_speed}m/s"
                    
                    # Cache the result
                    _weather_cache[location_key] = (weather_result, current_time)
                    
                    print(f"✅ Real weather data from OpenWeatherMap for {location}")
                    return weather_result
                    
                elif response.status == 401:
                    return "Weather data unavailable - invalid API key"
                else:
                    return f"Weather data unavailable - API error {response.status}"
                    
    except Exception as e:
        print(f"⚠️ Weather API error: {e}")
        return f"Weather data unavailable - {str(e)}"


@function_tool(strict_mode=False) 
async def search_web(query: str, num_results: int = 5) -> List[SearchResult]:
    """
    Search the web for activities using Firecrawl API.
    
    Args:
        query: Search query string (e.g., "best hiking trails San Francisco")
        num_results: Number of results to return (default 5, max 10)
        
    Returns:
        List of SearchResult objects with url, title, and snippet
    """
    try:
        from firecrawl import AsyncFirecrawlApp
        
        api_key = get_api_key("firecrawl")
        if not api_key:
            raise ValueError("Firecrawl API key not configured")
        
        firecrawl_app = AsyncFirecrawlApp(api_key=api_key)
        num_results = min(num_results, 10)  # Cap at 10 results
        
        print(f"🔍 Searching with Firecrawl: '{query}' (max {num_results} results)")
        
        response = await firecrawl_app.search(
            query=query,
            limit=num_results
        )
        
        # Handle different response formats
        data_items = []
        if hasattr(response, 'data') and response.data:
            data_items = response.data
        elif hasattr(response, 'results'):
            data_items = response.results
        elif isinstance(response, list):
            data_items = response
        elif isinstance(response, dict):
            data_items = response.get('data', response.get('results', []))
        
        results = []
        for i, item in enumerate(data_items[:num_results]):
            # Extract fields with fallbacks
            if isinstance(item, dict):
                url = item.get('url', item.get('link', ''))
                title = item.get('title', item.get('name', f'Result {i+1}'))
                snippet = item.get('description', item.get('snippet', item.get('content', '')))
            else:
                url = getattr(item, 'url', getattr(item, 'link', ''))
                title = getattr(item, 'title', getattr(item, 'name', f'Result {i+1}'))
                snippet = getattr(item, 'description', getattr(item, 'snippet', getattr(item, 'content', '')))
            
            # Truncate snippet if too long
            if len(snippet) > 200:
                snippet = snippet[:200] + "..."
            
            if url:  # Only include results with valid URLs
                results.append(SearchResult(
                    url=url,
                    title=title,
                    snippet=snippet
                ))
        
        print(f"✅ Found {len(results)} search results")
        return results
        
    except Exception as e:
        print(f"❌ Search error: {e}")
        return [SearchResult(url="", title="Search failed", snippet=f"Error: {e}")]


@function_tool(strict_mode=False)
async def scrape_results(url: str, user_intent: str, max_retries: int = 2) -> ExtractedActivity:
    """
    Smart scraping that handles both individual activity pages and list pages properly.
    
    NEW APPROACH:
    1. Analyze page type (individual activity vs list vs mixed)
    2. For individual pages: Extract directly
    3. For list pages: Find best matching sub-activity and follow its URL
    4. Ensure metrics belong to a single, specific activity
    
    Args:
        url: The URL to scrape and extract from
        user_intent: User's activity intent for relevance scoring (JSON string)
        max_retries: Maximum retry attempts (default 2)
        
    Returns:
        ExtractedActivity object with accurate, non-mixed activity data
    """
    try:
        from firecrawl import AsyncFirecrawlApp
        
        api_key = get_api_key("firecrawl")
        if not api_key:
            return _create_error_activity(url, "Firecrawl API key not configured")
        
        firecrawl_app = AsyncFirecrawlApp(api_key=api_key)
        
        print(f"🧠 Starting smart scraping for: {url}")
        
        # Step 1: Scrape the initial page
        content = await _scrape_with_retries(firecrawl_app, url, max_retries)
        if not content:
            return _create_error_activity(url, "Failed to scrape content")
        
        # Step 2: Analyze the page to determine its type and structure
        smart_result = await _smart_page_analysis(content, user_intent, url)
        
        if not smart_result.success:
            print(f"❌ Smart analysis failed")
            return _create_error_activity(url, "Page analysis failed")
        
        # Step 3: Handle different page types with appropriate strategies
        if smart_result.extraction_strategy == "direct":
            # Individual activity page - extract directly
            print(f"✅ Individual activity page - extracting directly")
            return smart_result.extracted_activity
            
        elif smart_result.extraction_strategy == "sub_page_follow":
            # List page with clear best match - follow the sub-URL
            print(f"🔗 List page detected - following best sub-URL: {smart_result.follow_up_url}")
            return await _scrape_sub_page(firecrawl_app, smart_result.follow_up_url, user_intent, max_retries)
            
        elif smart_result.extraction_strategy == "list_selection":
            # List page - select best candidate from available options
            print(f"📋 List page - selecting best candidate from {len(smart_result.candidate_activities)} options")
            return await _extract_from_best_candidate(smart_result.candidate_activities, content, user_intent, url)
            
        else:
            print(f"❌ No clear extraction strategy available")
            return _create_error_activity(url, "Could not determine how to extract activity data")
        
    except Exception as e:
        print(f"❌ Smart scraping error for {url}: {e}")
        return _create_error_activity(url, str(e))


@function_tool(strict_mode=False)
async def analyze_feedback(user_feedback: str, conversation_context: str) -> TurnFeedback:
    """
    Flexible feedback analyzer that asks LLM to clarify user intent using examples.
    
    Instead of pattern matching, sends feedback to LLM with clear examples
    to determine what action to take next.
    
    Args:
        user_feedback: The user's natural language feedback
        conversation_context: JSON string with conversation history and context
        
    Returns:
        TurnFeedback object with structured feedback analysis
    """
    try:
        print(f"🧠 Asking LLM to clarify feedback: '{user_feedback[:50]}...'")
        
        client = get_instructor_client()
        config = get_config()
        
        prompt = f"""
        The user provided feedback on activity recommendations. Please analyze their intent and tell me what action to take.

        USER FEEDBACK: "{user_feedback}"

        EXAMPLES OF DIFFERENT FEEDBACK TYPES:

        SATISFIED (user is happy, conversation can end):
        - "Perfect, thanks!"
        - "These look great"
        - "That's exactly what I wanted"
        - "Good suggestions"
        - "I like these"

        NEW_SEARCH (user wants completely different activities, need new search):
        - "I'd rather do something indoors"
        - "Actually, I prefer cycling instead of hiking"
        - "Can you find something easier?"
        - "What about water sports instead?"
        - "I want something closer to downtown"
        - "How about group activities?"

        REFINEMENT (user likes current type but wants small changes):
        - "These are too difficult"
        - "Make it shorter duration"
        - "Something with less equipment needed"
        - "A bit closer to my location"

        UNCLEAR (cannot determine intent, need clarification):
        - "Hmm"
        - "Maybe"
        - "I don't know"
        - Very short or ambiguous responses

        Based on the user's feedback above, classify it as one of: "satisfied", "new_search", "refinement", or "unclear"

        If it's "new_search", extract what new type of activity they want.
        If it's "refinement", extract what specific changes they want.
        """
        
        feedback_analysis = await client.chat.completions.create(
            model=config.models.analysis_model.replace("openai/", ""),
            messages=[
                {"role": "system", "content": "You are an expert at understanding user intent from feedback. Classify the feedback accurately based on the examples provided."},
                {"role": "user", "content": prompt}
            ],
            response_model=TurnFeedback,
            temperature=0.1,
            max_tokens=300
        )
        
        print(f"✅ LLM classified feedback as: {feedback_analysis.feedback_status}")
        return feedback_analysis
        
    except Exception as e:
        print(f"❌ Feedback analysis error: {e}")
        return TurnFeedback(
            conversation_id="error",
            turn_number=0,
            user_input=user_feedback,
            system_recommendations=[],
            user_feedback=user_feedback,
            feedback_status="unclear",
            extracted_updates={"error": str(e)},
            timestamp=""
        )


# --- New Smart Scraping Helper Functions ---

async def _smart_page_analysis(content: str, user_intent: str, url: str):
    """Analyze page content to determine the best extraction strategy."""
    
    try:
        from core.models import SmartExtractionResult, PageAnalysis, ActivityCandidate
        
        client = get_instructor_client()
        config = get_config()
        
        # Truncate content if too long
        if len(content) > config.system.max_content_chars:
            content = content[:config.system.max_content_chars] + "...[truncated]"
        
        prompt = f"""
        Analyze this web page to determine the best strategy for extracting activity data.
        
        USER INTENT: {user_intent}
        SOURCE URL: {url}
        CONTENT: {content}
        
        DETERMINE PAGE TYPE:
        1. INDIVIDUAL_ACTIVITY: Page about one specific activity/trail/route with detailed info
           - Examples: Single trail page, specific route details, one gym class
           - Usually has detailed metrics for that one activity
           
        2. ACTIVITY_LIST: Page listing multiple activities/trails/routes 
           - Examples: "Best 10 trails in Vienna", "Running routes near you", "AllTrails category page"
           - Contains multiple distinct activities
           - May have links to individual activity pages
           
        3. MIXED_CONTENT: Page with general info, maybe some activities but not focused
           - Examples: Blog posts, general location guides, mixed content
        
        EXTRACTION STRATEGY:
        - DIRECT: Individual activity page - extract the single activity directly
        - SUB_PAGE_FOLLOW: List page with clear sub-URLs - follow the best matching sub-URL
        - LIST_SELECTION: List page without sub-URLs - select best candidate from list content
        - FAILED: Cannot extract meaningful activity data
        
        For LIST pages, identify:
        1. Individual activities mentioned (with names, brief descriptions)
        2. Any sub-URLs to detailed pages for those activities
        3. Which activity best matches the user intent
        
        For INDIVIDUAL pages, check if there's enough detail for direct extraction.
        
        Be VERY careful about data mixing - if this is a list page, DO NOT try to extract
        metrics that might belong to different activities.
        """
        
        # First, get page analysis
        page_analysis = await client.chat.completions.create(
            model=config.models.scraper_model.replace("openai/", ""),
            messages=[
                {"role": "system", "content": "You are an expert at analyzing web page structure and content for activity extraction."},
                {"role": "user", "content": prompt}
            ],
            response_model=PageAnalysis,
            temperature=0.1,
            max_tokens=800
        )
        
        print(f"📊 Page analysis: {page_analysis.page_type}, {page_analysis.activity_count} activities, confidence: {page_analysis.confidence:.2f}")
        
        # Handle different page types
        if page_analysis.page_type == "individual_activity" and page_analysis.confidence > 0.6:
            # Direct extraction from individual page
            extracted_activity = await _extract_activity_with_instructor(content, user_intent, url)
            
            return SmartExtractionResult(
                success=True,
                page_analysis=page_analysis,
                extracted_activity=extracted_activity,
                extraction_strategy="direct"
            )
            
        elif page_analysis.page_type == "activity_list" and page_analysis.activity_count > 1:
            # List page - find candidates and determine follow-up strategy
            
            if page_analysis.sub_urls and page_analysis.best_match_url:
                # Has sub-URLs - follow the best one
                return SmartExtractionResult(
                    success=True,
                    page_analysis=page_analysis,
                    follow_up_url=page_analysis.best_match_url,
                    extraction_strategy="sub_page_follow"
                )
            else:
                # No sub-URLs - extract candidates from list content
                candidates = await _extract_activity_candidates(content, user_intent, url)
                
                return SmartExtractionResult(
                    success=True,
                    page_analysis=page_analysis,
                    candidate_activities=candidates,
                    extraction_strategy="list_selection"
                )
        else:
            # Mixed content or low confidence - try basic extraction
            extracted_activity = await _extract_activity_with_instructor(content, user_intent, url)
            
            if extracted_activity.relevance_score > 0.3:
                return SmartExtractionResult(
                    success=True,
                    page_analysis=page_analysis,
                    extracted_activity=extracted_activity,
                    extraction_strategy="direct"
                )
            else:
                return SmartExtractionResult(
                    success=False,
                    page_analysis=page_analysis,
                    extraction_strategy="failed"
                )
        
    except Exception as e:
        print(f"⚠️ Page analysis error: {e}")
        # Fallback to basic extraction
        try:
            extracted_activity = await _extract_activity_with_instructor(content, user_intent, url)
            return SmartExtractionResult(
                success=True,
                page_analysis=PageAnalysis(
                    page_type="mixed_content",
                    has_multiple_activities=False,
                    activity_count=1,
                    has_detailed_metrics=False,
                    confidence=0.3
                ),
                extracted_activity=extracted_activity,
                extraction_strategy="direct"
            )
        except:
            return SmartExtractionResult(
                success=False,
                page_analysis=PageAnalysis(
                    page_type="mixed_content",
                    has_multiple_activities=False,
                    activity_count=0,
                    has_detailed_metrics=False,
                    confidence=0.0
                ),
                extraction_strategy="failed"
            )


async def _extract_activity_candidates(content: str, user_intent: str, url: str):
    """Extract activity candidates from a list page."""
    
    try:
        from core.models import ActivityCandidate
        
        client = get_instructor_client()
        config = get_config()
        
        prompt = f"""
        Extract individual activity candidates from this list page content.
        
        USER INTENT: {user_intent}
        CONTENT: {content}
        
        Find distinct, individual activities mentioned on this page:
        - Look for specific trail names, route names, activity titles
        - Extract brief descriptions for each
        - Score relevance to user intent (0.0-1.0)
        - Note if any have detailed metrics available
        - Identify any sub-URLs to detailed pages (if links are present)
        
        Examples of good candidates:
        - "Prater Hauptallee Loop - 4.2km easy running path"
        - "Schönbrunn Palace Run - moderate difficulty palace grounds route"
        - "Donauinsel Long Trail - 21km flat running path along Danube"
        
        Return up to 5 best candidates, ranked by relevance to user intent.
        """
        
        # Note: This is a simplified approach - in production you'd want a more structured extraction
        candidates = await client.chat.completions.create(
            model=config.models.scraper_model.replace("openai/", ""),
            messages=[
                {"role": "system", "content": "Extract individual activity candidates from list content."},
                {"role": "user", "content": prompt}
            ],
            response_model=List[ActivityCandidate],
            temperature=0.1,
            max_tokens=600
        )
        
        print(f"🎯 Found {len(candidates)} activity candidates")
        return candidates[:5]  # Limit to top 5
        
    except Exception as e:
        print(f"⚠️ Candidate extraction error: {e}")
        return []


async def _scrape_sub_page(firecrawl_app, sub_url: str, user_intent: str, max_retries: int) -> ExtractedActivity:
    """Scrape a specific sub-page for detailed activity information."""
    
    try:
        print(f"🔗 Following sub-page: {sub_url}")
        
        # Scrape the sub-page
        sub_content = await _scrape_with_retries(firecrawl_app, sub_url, max_retries)
        if not sub_content:
            return _create_error_activity(sub_url, "Failed to scrape sub-page content")
        
        # Extract from the specific page
        extracted_activity = await _extract_activity_with_instructor(sub_content, user_intent, sub_url)
        
        print(f"✅ Sub-page extraction: {extracted_activity.activity_name} (relevance: {extracted_activity.relevance_score:.2f})")
        return extracted_activity
        
    except Exception as e:
        print(f"❌ Sub-page scraping error: {e}")
        return _create_error_activity(sub_url, f"Sub-page extraction failed: {e}")


async def _extract_from_best_candidate(candidates, content: str, user_intent: str, url: str) -> ExtractedActivity:
    """Extract detailed activity from the best candidate in a list."""
    
    if not candidates:
        return _create_error_activity(url, "No suitable activity candidates found")
    
    # Sort by relevance and take the best
    best_candidate = max(candidates, key=lambda c: c.relevance_score)
    
    print(f"🎯 Selected best candidate: {best_candidate.activity_name} (relevance: {best_candidate.relevance_score:.2f})")
    
    try:
        client = get_instructor_client()
        config = get_config()
        
        # Focus extraction on the specific activity
        focused_prompt = f"""
        Extract detailed information for this SPECIFIC activity from the content:
        
        TARGET ACTIVITY: {best_candidate.activity_name}
        ACTIVITY DESCRIPTION: {best_candidate.brief_description}
        USER INTENT: {user_intent}
        SOURCE URL: {url}
        FULL CONTENT: {content}
        
        FOCUS ONLY ON THE TARGET ACTIVITY ABOVE.
        Do not mix information from other activities mentioned in the content.
        
        Extract detailed metrics ONLY if they clearly belong to "{best_candidate.activity_name}":
        - Distance, elevation, time, rating, etc.
        - If metrics are ambiguous or could belong to multiple activities, leave them as None
        - Be extremely conservative - better to miss details than to mix data
        
        Set details_available=True ONLY if you find specific metrics for this exact activity.
        """
        
        extracted_activity = await client.chat.completions.create(
            model=config.models.scraper_model.replace("openai/", ""),
            messages=[
                {"role": "system", "content": "Extract information for the specific target activity only. Do not mix data from multiple activities."},
                {"role": "user", "content": focused_prompt}
            ],
            response_model=ExtractedActivity,
            temperature=0.1,
            max_tokens=config.models.max_tokens
        )
        
        return extracted_activity
        
    except Exception as e:
        print(f"⚠️ Focused extraction error: {e}")
        return _create_error_activity(url, f"Focused extraction failed: {e}")


# --- Internal Helper Functions ---

async def _scrape_with_retries(firecrawl_app, url: str, max_retries: int) -> Optional[str]:
    """Scrape content from URL with retries and timeout handling."""
    
    # Skip problematic domains
    problematic_domains = ['facebook.com', 'reddit.com', 'instagram.com', 'twitter.com', 'x.com', 'youtube.com', 'youtu.be']
    if any(domain in url.lower() for domain in problematic_domains):
        print(f"🚫 Skipping problematic domain: {url}")
        return None
    
    config = get_config()
    
    # Reduce retries to 1 to avoid excessive retrying
    max_retries = min(max_retries, 1)
    
    for attempt in range(max_retries + 1):
        try:
            attempt_text = f" (attempt {attempt + 1}/{max_retries + 1})" if attempt > 0 else ""
            print(f"🕷️ Scraping with Firecrawl: {url}{attempt_text}")
            
            # Add timeout
            response = await asyncio.wait_for(
                firecrawl_app.scrape_url(
                    url=url,
                    formats=['markdown'],
                    only_main_content=True
                ),
                timeout=config.system.scrape_timeout
            )
            
            # Extract content from ScrapeResponse object
            content = ""
            if hasattr(response, 'data'):
                data = response.data
                
                if hasattr(data, 'markdown') and data.markdown:
                    content = data.markdown
                elif hasattr(data, 'content') and data.content:
                    content = data.content
                elif hasattr(data, 'text') and data.text:
                    content = data.text
                elif isinstance(data, str):
                    content = data
                elif isinstance(data, dict):
                    content = data.get('markdown', '') or data.get('content', '') or data.get('text', '')
            
            # Fallback: try direct access
            if not content and hasattr(response, 'markdown'):
                content = response.markdown
            elif not content and hasattr(response, 'content'):
                content = response.content
            elif not content and isinstance(response, dict):
                content = response.get('data', {}).get('markdown', '') or response.get('data', {}).get('content', '')
            
            if content and len(content.strip()) > 100:
                print(f"✅ Successfully scraped {len(content)} characters from {url}")
                return content
            else:
                if attempt < max_retries:
                    print(f"⚠️ No meaningful content, retrying in 2 seconds...")
                    await asyncio.sleep(2)
                    continue
                else:
                    print(f"❌ No meaningful content found after {max_retries + 1} attempts")
                    return None
                    
        except asyncio.TimeoutError:
            if attempt < max_retries:
                print(f"⏰ Timeout, retrying in 3 seconds...")
                await asyncio.sleep(3)
                continue
            else:
                print(f"❌ Timeout after {max_retries + 1} attempts")
                return None
        except Exception as e:
            if attempt < max_retries:
                print(f"⚠️ Error: {e}, retrying in 2 seconds...")
                await asyncio.sleep(2)
                continue
            else:
                print(f"❌ Failed after {max_retries + 1} attempts: {e}")
                return None
    
    return None


async def _smart_page_analysis(content: str, user_intent: str, url: str) -> "SmartExtractionResult":
    """Analyze page content to determine the best extraction strategy."""
    
    try:
        from core.models import SmartExtractionResult, PageAnalysis, ActivityCandidate
        
        client = get_instructor_client()
        config = get_config()
        
        # Truncate content if too long
        if len(content) > config.system.max_content_chars:
            content = content[:config.system.max_content_chars] + "...[truncated]"
        
        prompt = f"""
        Analyze this web page to determine the best strategy for extracting activity data.
        
        USER INTENT: {user_intent}
        SOURCE URL: {url}
        CONTENT: {content}
        
        DETERMINE PAGE TYPE:
        1. INDIVIDUAL_ACTIVITY: Page about one specific activity/trail/route with detailed info
           - Examples: Single trail page, specific route details, one gym class
           - Usually has detailed metrics for that one activity
           
        2. ACTIVITY_LIST: Page listing multiple activities/trails/routes 
           - Examples: "Best 10 trails in Vienna", "Running routes near you", "AllTrails category page"
           - Contains multiple distinct activities
           - May have links to individual activity pages
           
        3. MIXED_CONTENT: Page with general info, maybe some activities but not focused
           - Examples: Blog posts, general location guides, mixed content
        
        EXTRACTION STRATEGY:
        - DIRECT: Individual activity page - extract the single activity directly
        - SUB_PAGE_FOLLOW: List page with clear sub-URLs - follow the best matching sub-URL
        - LIST_SELECTION: List page without sub-URLs - select best candidate from list content
        - FAILED: Cannot extract meaningful activity data
        
        For LIST pages, identify:
        1. Individual activities mentioned (with names, brief descriptions)
        2. Any sub-URLs to detailed pages for those activities
        3. Which activity best matches the user intent
        
        For INDIVIDUAL pages, check if there's enough detail for direct extraction.
        
        Be VERY careful about data mixing - if this is a list page, DO NOT try to extract
        metrics that might belong to different activities.
        """
        
        # First, get page analysis
        page_analysis = await client.chat.completions.create(
            model=config.models.scraper_model.replace("openai/", ""),
            messages=[
                {"role": "system", "content": "You are an expert at analyzing web page structure and content for activity extraction."},
                {"role": "user", "content": prompt}
            ],
            response_model=PageAnalysis,
            temperature=0.1,
            max_tokens=800
        )
        
        print(f"📊 Page analysis: {page_analysis.page_type}, {page_analysis.activity_count} activities, confidence: {page_analysis.confidence:.2f}")
        
        # Handle different page types
        if page_analysis.page_type == "individual_activity" and page_analysis.confidence > 0.6:
            # Direct extraction from individual page
            extracted_activity = await _extract_activity_with_instructor(content, user_intent, url)
            
            return SmartExtractionResult(
                success=True,
                page_analysis=page_analysis,
                extracted_activity=extracted_activity,
                extraction_strategy="direct"
            )
            
        elif page_analysis.page_type == "activity_list" and page_analysis.activity_count > 1:
            # List page - find candidates and determine follow-up strategy
            
            if page_analysis.sub_urls and page_analysis.best_match_url:
                # Has sub-URLs - follow the best one
                return SmartExtractionResult(
                    success=True,
                    page_analysis=page_analysis,
                    follow_up_url=page_analysis.best_match_url,
                    extraction_strategy="sub_page_follow"
                )
            else:
                # No sub-URLs - extract candidates from list content
                candidates = await _extract_activity_candidates(content, user_intent, url)
                
                return SmartExtractionResult(
                    success=True,
                    page_analysis=page_analysis,
                    candidate_activities=candidates,
                    extraction_strategy="list_selection"
                )
        else:
            # Mixed content or low confidence - try basic extraction
            extracted_activity = await _extract_activity_with_instructor(content, user_intent, url)
            
            if extracted_activity.relevance_score > 0.3:
                return SmartExtractionResult(
                    success=True,
                    page_analysis=page_analysis,
                    extracted_activity=extracted_activity,
                    extraction_strategy="direct"
                )
            else:
                return SmartExtractionResult(
                    success=False,
                    page_analysis=page_analysis,
                    extraction_strategy="failed"
                )
        
    except Exception as e:
        print(f"⚠️ Page analysis error: {e}")
        # Fallback to basic extraction
        try:
            extracted_activity = await _extract_activity_with_instructor(content, user_intent, url)
            return SmartExtractionResult(
                success=True,
                page_analysis=PageAnalysis(
                    page_type="mixed_content",
                    has_multiple_activities=False,
                    activity_count=1,
                    has_detailed_metrics=False,
                    confidence=0.3
                ),
                extracted_activity=extracted_activity,
                extraction_strategy="direct"
            )
        except:
            return SmartExtractionResult(
                success=False,
                page_analysis=PageAnalysis(
                    page_type="mixed_content",
                    has_multiple_activities=False,
                    activity_count=0,
                    has_detailed_metrics=False,
                    confidence=0.0
                ),
                extraction_strategy="failed"
            )


async def _extract_from_best_candidate(candidates: List["ActivityCandidate"], content: str, user_intent: str, url: str) -> ExtractedActivity:
    """Extract detailed activity from the best candidate in a list."""
    
    if not candidates:
        return _create_error_activity(url, "No suitable activity candidates found")
    
    # Sort by relevance and take the best
    best_candidate = max(candidates, key=lambda c: c.relevance_score)
    
    print(f"🎯 Selected best candidate: {best_candidate.activity_name} (relevance: {best_candidate.relevance_score:.2f})")
    
    try:
        client = get_instructor_client()
        config = get_config()
        
        # Focus extraction on the specific activity
        focused_prompt = f"""
        Extract detailed information for this SPECIFIC activity from the content:
        
        TARGET ACTIVITY: {best_candidate.activity_name}
        ACTIVITY DESCRIPTION: {best_candidate.brief_description}
        USER INTENT: {user_intent}
        SOURCE URL: {url}
        FULL CONTENT: {content}
        
        FOCUS ONLY ON THE TARGET ACTIVITY ABOVE.
        Do not mix information from other activities mentioned in the content.
        
        Extract detailed metrics ONLY if they clearly belong to "{best_candidate.activity_name}":
        - Distance, elevation, time, rating, etc.
        - If metrics are ambiguous or could belong to multiple activities, leave them as None
        - Be extremely conservative - better to miss details than to mix data
        
        Set details_available=True ONLY if you find specific metrics for this exact activity.
        """
        
        extracted_activity = await client.chat.completions.create(
            model=config.models.scraper_model.replace("openai/", ""),
            messages=[
                {"role": "system", "content": "Extract information for the specific target activity only. Do not mix data from multiple activities."},
                {"role": "user", "content": focused_prompt}
            ],
            response_model=ExtractedActivity,
            temperature=0.1,
            max_tokens=config.models.max_tokens
        )
        
        return extracted_activity
        
    except Exception as e:
        print(f"⚠️ Focused extraction error: {e}")
        return _create_error_activity(url, f"Focused extraction failed: {e}")


async def _extract_activity_with_instructor(content: str, user_intent: str, url: str) -> ExtractedActivity:
    """Extract structured activity data using Instructor for type safety with anti-mixing safeguards."""
    
    try:
        client = get_instructor_client()
        config = get_config()
        
        # Truncate content if too long
        if len(content) > config.system.max_content_chars:
            content = content[:config.system.max_content_chars] + "...[truncated]"
        
        prompt = f"""
        Extract SPECIFIC activity information from this web content. Focus on individual, concrete activities rather than general categories.
        
        USER INTENT: {user_intent}
        SOURCE URL: {url}
        CONTENT: {content}
        
        CRITICAL WARNING ABOUT DATA MIXING:
        If this page contains MULTIPLE activities/trails (e.g., "Best 10 trails", "Running routes list", "AllTrails category page"):
        - DO NOT mix metrics from different activities
        - If you see metrics for multiple activities, leave all detail fields as None
        - Only extract metrics if they clearly belong to ONE specific activity
        - Set details_available=False for list pages unless metrics are clearly for one activity
        
        EXTRACT SPECIFIC ACTIVITIES:
        - Look for named trails, routes, locations, or specific activities
        - Examples: "Donauinsel 5km Loop", "Prater Park Morning Run", "Schönbrunn Palace Garden Walk"
        - NOT: "Running Routes in Vienna" or "General Hiking Options"
        - If multiple specific activities are mentioned, choose the MOST specific one AND warn about data mixing
        
        DETAILED METRICS (ONLY if explicitly mentioned AND belong to ONE activity):
        - distance: Extract exact distance/length if mentioned (e.g., "3.9 mi", "6.3 km", "5 kilometers")
        - elevation_gain: Extract elevation gain if mentioned (e.g., "301 ft", "91.7 m", "500 feet gain")
        - estimated_time: Extract completion time if mentioned (e.g., "1 hr 25 min", "45 minutes", "2 hours")
        - average_rating: Extract user ratings if mentioned (e.g., "4.6/5 stars", "4.2 out of 5")
        - surface_type: Extract surface if mentioned (e.g., "paved paths", "trail", "mixed terrain")
        - starting_point: Extract specific start location if mentioned
        - route_type: Extract route type if mentioned (e.g., "loop", "out-and-back", "point-to-point")
        
        ULTRA-CRITICAL RULES FOR DETAILED METRICS:
        1. ONLY extract if explicitly stated AND clearly belongs to ONE specific activity
        2. If this is a list page with multiple activities, set ALL detail fields to None
        3. Use exact wording from source for measurements
        4. Set details_available=True ONLY if at least 2 detailed metrics belong to the SAME activity
        5. If information is not available or ambiguous, leave fields as None - DO NOT GUESS
        6. Be ultra-conservative - better to miss details than mix data from different activities
        
        Extract:
        1. SPECIFIC activity name (not generic categories)
        2. Exact location with details
        3. Detailed description with specifics (route details, landmarks, terrain)
        4. Basic info: difficulty, duration, equipment
        5. Weather suitability and indoor/outdoor type
        6. Detailed metrics ONLY if available for ONE activity
        7. Relevance score to user intent (0.0-1.0, be strict)
        8. Confidence in extracted data
        
        RELEVANCE SCORING (be strict):
        - 1.0: Perfect match - exact specific activity, location, preferences
        - 0.8-0.9: Very good - specific activity of right type, good location match
        - 0.6-0.7: Good - specific activity, related location or type
        - 0.4-0.5: Partial - somewhat specific, loosely related
        - 0.2-0.3: Weak - generic or barely related
        - 0.0-0.1: No match - unrelated or too generic
        """
        
        extracted_activity = await client.chat.completions.create(
            model=config.models.scraper_model.replace("openai/", ""),
            messages=[
                {"role": "system", "content": "You are an expert at extracting structured activity information from web content. NEVER mix data from different activities."},
                {"role": "user", "content": prompt}
            ],
            response_model=ExtractedActivity,
            temperature=config.models.temperature,
            max_tokens=config.models.max_tokens
        )
        
        return extracted_activity
        
    except Exception as e:
        print(f"⚠️ Instructor extraction error: {e}")
        return _create_error_activity(url, f"Extraction failed: {e}")


def _create_error_activity(url: str, error_msg: str) -> ExtractedActivity:
    """Create an error ExtractedActivity object."""
    return ExtractedActivity(
        source_url=url,
        activity_name="Extraction Failed",
        location="Unknown",
        description=f"Failed to extract activity data: {error_msg}",
        difficulty_level="unknown",
        duration_estimate="unknown",
        equipment_needed=[],
        weather_suitability="unknown",
        indoor_outdoor="unknown",
        relevance_score=0.0,
        extraction_confidence="low"
    )
