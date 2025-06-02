# Vibe Shopping Agent

A conversational shopping assistant that recommends clothing items based on vibes and preferences while maintaining a natural conversation flow.

## Features

- **Vibe-based Recommendations**: Get personalized clothing recommendations based on described vibes or styles
- **Contextual Understanding**: The agent understands and responds naturally to user preferences
- **Smart Follow-ups**: Limited to 2 follow-up questions to gather necessary information
- **Budget Filtering**: Apply budget constraints to recommendations at any point in the conversation
- **Conversation Reset**: Type "reset" to start a new shopping conversation

## Project Structure

```
vibe-shopping/
│
├── backend/                    # Server-side code
│   ├── app.py                  # Flask application entry point
│   ├── agent.py                # Shopping agent implementation
│   ├── dynamic_keyword_mapper.py # Maps user queries to keywords
│   ├── advanced_enhancements.py # Enhanced filtering and recommendations
│   ├── keyword_mappings.py     # Base keyword mapping definitions
│   ├── vibe_to_attribute.txt   # Mapping from vibes to attributes
│   ├── Apparels_shared.xlsx    # Clothing item database (Excel)
│   ├── .env                    # Environment configuration
│   └── requirements.txt        # Python dependencies
│
└── frontend/                   # Client-side code
    ├── templates/              # HTML templates
    │   └── index.html          # Main application page
    └── static/                 # Static assets
        ├── script.js           # Frontend JavaScript
        └── style.css           # CSS styles
```

## Setup and Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   cd vibe-shopping/backend
   pip install -r requirements.txt
   ```
3. Set up your .env file with required API keys
4. Run the application:
   ```bash
   cd vibe-shopping/backend
   python app.py
   ```
5. Open your browser and navigate to http://localhost:5001

## API Endpoints

- `GET /`: Main application page
- `POST /chat`: Process shopping queries
- `GET /health`: Check application health

## Conversation Flow

1. User describes a vibe, style, or specific clothing item they're looking for
2. Agent provides recommendations based on user input
3. Agent may ask up to 2 follow-up questions to refine recommendations
4. User can refine results by mentioning budget constraints or additional preferences
5. Type "reset" to start a new conversation

## Recent Fixes & Improvements

### ✅ Consistency Issues Resolved (June 2025)

**Problem**: The system was showing different numbers of recommendations (e.g., 4 vs 7 items) for the same query with identical filters, leading to inconsistent user experience.

**Root Cause**: 
- Non-deterministic sampling logic in `generate_enhanced_recommendations()` function
- Additional trending items being injected randomly
- Complex multi-step processing with varying fallback mechanisms

**Solution Applied**:
1. **Deterministic Selection**: Replaced complex sampling with consistent top-N selection
2. **Removed Random Elements**: Eliminated trending item injection that caused count variations  
3. **Accurate Justification**: Fixed justification messages to reflect actual final recommendation count
4. **Consistent Processing**: Streamlined recommendation generation to always produce same results for same input

**Result**: Same query now consistently returns same number of recommendations with accurate justification messages.

### ✅ Previous Fixes Maintained

- **Excel Migration**: Successfully switched from CSV to Excel file handling
- **JSON Syntax**: Fixed vibe mapping parsing errors (now loads 11 mappings vs 8)
- **Catalog Compatibility**: Smart AI-powered validation blocks unavailable items (coats, shoes, etc.)
- **Process Flow**: All queries now check catalog compatibility and reset state for incompatible requests

## Enhancement Roadmap

### 🚀 Suggested Next Phase Enhancements

#### 1. **Advanced Conversation Memory**
- **Goal**: Remember user preferences across sessions
- **Implementation**: Add user preference storage and retrieval
- **Benefit**: Personalized shopping experience with remembered size, budget, style preferences

#### 2. **Smart Follow-up Question Prioritization**
- **Goal**: Ask the most relevant questions first based on query context
- **Implementation**: Enhance question prioritization algorithm
- **Benefit**: Faster path to relevant recommendations

#### 3. **Multi-item Request Handling**
- **Goal**: Handle requests like "I need a top and pants for work"
- **Implementation**: Parse and handle multiple product categories in one request
- **Benefit**: Support for complete outfit planning

#### 4. **Seasonal Recommendation Intelligence**
- **Goal**: Automatically suggest seasonally appropriate items
- **Implementation**: Enhance seasonal awareness and automatic seasonal filtering
- **Benefit**: More relevant recommendations based on current season and weather

#### 5. **Advanced Price Intelligence**
- **Goal**: Better budget understanding and price-based recommendations
- **Implementation**: 
  - Detect relative price terms ("affordable", "budget-friendly", "luxury")
  - Suggest items slightly above/below budget with explanations
  - Price trend awareness
- **Benefit**: More nuanced budget handling and price-conscious recommendations

#### 6. **Style Combination Understanding**
- **Goal**: Better understanding of complex style requests
- **Implementation**: Enhanced vibe combination detection (e.g., "boho chic", "minimalist professional")
- **Benefit**: More accurate style matching for sophisticated fashion requests

### 🎯 Technical Implementation Priorities

#### High Priority (Easy Wins)
1. **Add conversation memory**: Store user preferences in session
2. **Enhance seasonal intelligence**: Automatically factor current season into recommendations
3. **Improve price intelligence**: Better handling of relative price terms

#### Medium Priority (Moderate Effort)
1. **Multi-item request handling**: Support outfit planning
2. **Advanced style combination understanding**: Better vibe matching
3. **Enhanced error recovery**: Better confusion detection and clarification

#### Long Term (Significant Effort)
1. **Visual integration**: Image-based queries
2. **Conversation analytics**: Learning from user interactions
3. **Performance optimization**: Large-scale improvements

### 📊 Current System Status

#### Strengths
- ✅ Robust conversation flow handling
- ✅ Comprehensive follow-up question system
- ✅ Strong catalog compatibility checking
- ✅ Good attribute extraction and filtering
- ✅ Guided response format handling

#### Areas for Future Enhancement
- 🔄 User preference memory across sessions
- 🔄 Advanced seasonal intelligence
- 🔄 Multi-item outfit planning
- 🔄 Visual shopping capabilities
- 🔄 Performance optimization
