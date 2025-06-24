from typing import List, Optional
from pydantic import BaseModel, Field

class UserIntent(BaseModel):
    """Enhanced user intent with structured search parameters"""
    activity_type: str
    location: str
    weather_condition: str
    search_query: str  # Generated search query for this intent
    preferences: List[str] = []
    search_radius_km: Optional[int] = Field(default=25, description="How far to search for activities from location")
    indoor_outdoor: str 
    
    # Enhanced structured parameters
    difficulty_preference: Optional[str] = Field(default=None, description="easy, moderate, hard, or challenging")
    duration_preference: Optional[str] = Field(default=None, description="short (<30min), medium (30-60min), long (1-3hr), full day")
    elevation_preference: Optional[str] = Field(default=None, description="flat, rolling, hilly, mountainous")
    surface_preference: Optional[str] = Field(default=None, description="pavement, trails, mixed")
    starting_point: Optional[str] = Field(default=None, description="specific starting location if mentioned")
    distance_preference: Optional[str] = Field(default=None, description="specific distance if mentioned (e.g., '5km', '10 miles')")
    
    # Query quality assessment
    is_generic: bool = Field(default=True, description="Whether the query lacks specific details")
    needs_clarification: bool = Field(default=False, description="Whether to ask user for more details")

class SearchResult(BaseModel):
    """Basic search result from Firecrawl - raw data before extraction"""
    url: str
    title: str
    snippet: str

class ExtractedActivity(BaseModel):
    """Enhanced activity data with detailed structured information when available"""
    source_url: str
    activity_name: str
    location: str
    description: str
    
    # Basic details (existing)
    difficulty_level: Optional[str] = None
    duration_estimate: Optional[str] = None
    equipment_needed: List[str] = []
    weather_suitability: Optional[str] = None
    indoor_outdoor: Optional[str] = None
    
    # Enhanced structured details (only if available from source - NO HALLUCINATION)
    distance: Optional[str] = Field(default=None, description="Distance/length if available (e.g., '3.9 mi', '6.3 km')")
    elevation_gain: Optional[str] = Field(default=None, description="Elevation gain if available (e.g., '301 ft', '91.7 m')")
    estimated_time: Optional[str] = Field(default=None, description="Completion time if available (e.g., '1 hr 25 min')")
    average_rating: Optional[str] = Field(default=None, description="User rating if available (e.g., '4.6/5 stars')")
    surface_type: Optional[str] = Field(default=None, description="Surface type if mentioned (pavement, trails, mixed)")
    starting_point: Optional[str] = Field(default=None, description="Specific starting location if available")
    route_type: Optional[str] = Field(default=None, description="Route type if available (loop, out-and-back, point-to-point)")
    
    # Intelligence and scoring
    relevance_score: float = Field(ge=0.0, le=1.0, description="Relevance to user intent (0.0-1.0)")
    extraction_confidence: str = Field(default="medium", description="Confidence in extracted data: high, medium, low")
    details_available: bool = Field(default=False, description="Whether detailed metrics are available from source")

class ActivityRecommendation(BaseModel):
    """Enhanced activity recommendation with detailed information when available"""
    activity_name: str
    location: str
    description: str
    difficulty_level: str
    duration_estimate: str
    equipment_needed: List[str] = []
    weather_suitability: str
    indoor_outdoor: str
    weather_recommendation: Optional[str] = Field(
        default=None, 
        description="Weather-based recommendation (e.g., 'Great weather for outdoor!', 'Consider indoor due to rain')"
    )
    source_url: str
    
    # Enhanced details (only if available from source)
    distance: Optional[str] = None
    elevation_gain: Optional[str] = None
    estimated_time: Optional[str] = None
    average_rating: Optional[str] = None
    surface_type: Optional[str] = None
    starting_point: Optional[str] = None
    route_type: Optional[str] = None

class UserPreferences(BaseModel):
    """Structured preferences - pure data model"""
    activity_type: str
    location: str  
    search_radius_km: int = 25
    difficulty_level: Optional[str] = None
    duration_preference: Optional[str] = None
    indoor_outdoor: str = "both"
    weather_preference: Optional[str] = None

class ConversationState(BaseModel):
    """Minimal conversation state for single conversation prototype"""
    initial_request: str
    preferences: UserPreferences
    turn_count: int = 0
    max_turns: int = 5

class TurnFeedback(BaseModel):
    """Rich feedback data for ML evaluation and system improvement"""
    conversation_id: str = Field(default="unknown")
    turn_number: int = Field(default=0)
    user_input: str = Field(default="")
    system_recommendations: List[ActivityRecommendation] = Field(default_factory=list)
    user_feedback: str
    feedback_status: str = Field(description="satisfied, unclear, or refinement", default="unclear")
    extracted_updates: dict = Field(default_factory=dict, description="What preferences changed")
    timestamp: str = Field(default="")

class ConversationalResponse(BaseModel):
    """Conversational response with recommendations - simplified for prototype"""
    recommendations: List[ActivityRecommendation]
    conversation_message: str  # Friendly message referencing previous context

class PageAnalysis(BaseModel):
    """Analysis of a scraped page to determine its type and content structure"""
    page_type: str = Field(description="individual_activity, activity_list, or mixed_content")
    has_multiple_activities: bool = Field(description="Whether page contains multiple distinct activities")
    activity_count: int = Field(description="Number of distinct activities found on page")
    has_detailed_metrics: bool = Field(description="Whether page has detailed activity metrics")
    confidence: float = Field(ge=0.0, le=1.0, description="Confidence in page analysis")
    
    # For list pages
    sub_urls: List[str] = Field(default_factory=list, description="URLs to individual activity pages if found")
    best_match_url: Optional[str] = Field(default=None, description="Most relevant sub-URL for user intent")
    
class ActivityCandidate(BaseModel):
    """A candidate activity found on a list page before detailed extraction"""
    activity_name: str
    brief_description: str
    sub_url: Optional[str] = Field(default=None, description="Link to detailed page if available")
    relevance_score: float = Field(ge=0.0, le=1.0, description="How well it matches user intent")
    has_details: bool = Field(default=False, description="Whether this candidate has detailed metrics")
    
class SmartExtractionResult(BaseModel):
    """Result of intelligent scraping that handles both list and individual pages"""
    success: bool
    page_analysis: PageAnalysis
    extracted_activity: Optional[ExtractedActivity] = None
    candidate_activities: List[ActivityCandidate] = Field(default_factory=list)
    follow_up_url: Optional[str] = Field(default=None, description="Best URL to scrape next for more details")
    extraction_strategy: str = Field(description="direct, sub_page_follow, list_selection, or failed")
