# Activity Recommendation Agent

**Multi-turn conversational agent for personalized activity recommendations with enhanced smart web scraping and detailed extraction.**


## ✨ Features

- 🌍 **Global Activity Search**: Find activities anywhere in the world
- 🌤️ **Real-time Weather Integration**: Uses OpenWeather API for context -> so it does not recommend a outdoor run during a thunderstorm :) 
- 🧠 **Smart Web Scraping**: Iterative page analysis that prevents data mix-up
- 🎯 **Accurate Data Extraction**: Individual activity pages vs. list page detection
- 🤖 **Type-safe Agent Pipeline**: Built with OpenAI Agents SDK and Pydantic models
- 💬 **Interactive Chat Interface**: Terminal chat experience
- 🧠 **Intelligent Feedback Processing**: LLM-powered feedback analysis with context preservation
- 🔄 **Adaptive Search**: Smart preference evolution maintains context while incorporating refinements

## Assignment Mapping

**Multi-turn Interaction & Tool Use:**
- Conversational agent with feedback loops (3-5 turns)
- 4 specialized tools: weather API, web search, smart content scraping, feedback analysis
- Advanced preference evolution: maintains context while incorporating user refinements

**Agent Scaffold:**
- Built using OpenAI Agents SDK with specialized agents:
  - `IntentAgent`: Parses requests + gets weather data with caching
  - `ExtractionAgent`: Smart web search + intelligent content scraping  
  - `ConversationAgent`: Response generation + feedback analysis

**Problem Identification & Adjustment:**
- **Fixed**: Preference evolution (preserves activity type/location while updating preferences)
- **Fixed**: Data mixing prevention (smart page analysis prevents mixing metrics from different activities)
- **Fixed**: Weather API caching (eliminates duplicate calls)
- **Fixed**: Environment loading optimization (single load with status tracking)
- **Enhanced**: Detailed metrics extraction with strict no-hallucination policy

**Test Prompts:** Comprehensive test set created and validated for accuracy and relevance
**Multi-model Testing:** Limited testing with alternative models (nano) showed inconsistent results - missing location restrictions (suggesting 2000km routes when asking for local Vienna activities) and obvious outputs (bike equipment: "a bicycle"). Reliable results currently only with GPT-4o-mini.

### ❌ **Not Yet Implemented**

**Reward Function:** No formal evaluation metrics implemented  
**Best-of-N Selection:** Not implemented
**Parallel/Multi-agent:** Single-threaded implementation

## 🚀 Evolving Web Scraper

### **Problem Solved: Data Mixing Prevention**

**Before:** System would extract mixed data from list pages:
```
❌ "Prater Park Trail - 5km, 50m elevation, 30 min"
   → Actually mixed data from multiple trails on the same page!
```

**After:** Smart page analysis and targeted extraction:
```
✅ Detects list pages → Follows sub-URLs → Extracts from individual pages
✅ Prevents data mixing → Accurate, single-activity metrics
```


## Architecture

```
User Request → Intent Parsing → Enhanced Web Search → Smart Page Analysis
     ↑                                                        ↓
User Feedback ← Preference Evolution ← Response Generation ← Targeted Extraction
                      ↑                                           ↓
                Context Preservation ← Detail Validation ← Sub-page Following
```

**Enhanced Pipeline:**
1. **Intent Parsing**: Extract structured parameters + weather with caching
2. **Smart Web Search**: Geographic-specific queries with enhancement
3. **Intelligent Page Analysis**: Classify page type and determine extraction strategy
4. **Targeted Content Extraction**: 
   - Individual pages: Direct detailed extraction
   - List pages: Sub-URL following or candidate selection
   - Mixed content: Conservative extraction
5. **Response Generation**: Rich display with verified metrics
6. **Smart Feedback Analysis**: Context-preserving preference evolution


**Context-Preserving Feedback Evolution:**
```
Turn 1: "running route in Vienna"
→ Gets: General Vienna running routes

Turn 2: "longer routes, more in outskirts"  
→ Classification: REFINEMENT (not new search)
→ Preserves: running, Vienna
→ Updates: duration (longer), location (outskirts)
→ New search: "long running routes Vienna outskirts"

Turn 3: "actually prefer trails over pavement"
→ Preserves: running, Vienna, outskirts, longer duration
→ Updates: surface preference (trails)
→ Result: "long trail running routes Vienna outskirts"
```

## Current Challenges & Limitations

**Preference Conflict Resolution:**
- User: "closer" then later "distance doesn't matter but want different terrain"
- **Status**: Basic conflict detection but no systematic resolution framework

**Recommendation Diversity:**
- Search engines may return similar results repeatedly  
- **Status**: No diversity injection mechanisms implemented

**Success Metrics Definition:**
- Conversation success measurement unclear
- **Status**: No formal evaluation framework beyond manual testing

**Scraping Scope Limitations:**
- Some sites block or limit scraping
- Dynamic content may not be captured
- **Status**: Works well for major activity sites but coverage varies

## Next Steps (Future Enhancements)

**Evaluation Framework:**
- Formal test prompt sets for different activity types and locations
- Reward functions: preference satisfaction, detail completeness, conversation flow
- A/B testing framework for different strategies
- Best-of-N selection with evaluation functions

**Handoffs and Guardrails: OpenAI Agents SDK**
- Multi-agent collaboration for complex queries with handoffs
- Restricting what the agent can answer (guardrails)

**Enhanced Context & Scheduling:**
- Time-of-day aware recommendations for scheduled activities (yoga sessions, boxing classes, gym hours)
- Integration with local business hours and class schedules
- Dynamic availability checking for time-sensitive activities
- Smart scheduling optimization based on user's available time windows

**Dynamic/Advanced Scraping:**
- Deep course schedule extraction from fitness studios, gyms, and activity providers
- Multi-level page navigation to find buried scheduling information
- Structured data extraction from complex timetables and booking systems
- Real-time availability parsing from dynamic content and booking platforms
- Smart form interaction for accessing protected schedule information
- Calendar integration and recurring event pattern recognition

**Game Integration:**
- Hook into gaming system where activities (running, cycling) generate XP based on player's remaining "game-energy"
- Smart XP optimization queries: "I have 2 hours and want to maximize my XP with the energy I have left, what are my options?"
- Activity difficulty/reward calculation based on player stats and energy levels
- Real-world activity tracking integration with game progression

## Tech Stack

- **Framework**: OpenAI Agents SDK with async execution
- **APIs**: OpenWeather (cached), Firecrawl (search + scraping)  
- **Models**: GPT-4o-mini (intent, extraction, conversation, analysis)
- **Language**: Python + asyncio with comprehensive error handling
- **Type Safety**: Pydantic + Instructor for structured outputs
- **Architecture**: Agent-based pipeline with intelligent routing
