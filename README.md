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
│   ├── apparels_shared.csv     # Clothing item database (CSV)
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
