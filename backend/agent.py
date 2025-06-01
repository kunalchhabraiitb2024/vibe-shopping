import json
import re
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from backend.keyword_mappings import get_keyword_attributes, search_keywords
from backend.dynamic_keyword_mapper import DynamicKeywordMapper, get_dynamic_keywords
from backend.advanced_enhancements import (
    AdvancedVibeEngine, SmartFiltering, ConversationFlowManager, 
    AdvancedRecommendationEngine,
    apply_advanced_filtering, generate_enhanced_recommendations
)

# Load environment variables
load_dotenv()

class VibeShoppingAgent:
    def __init__(self, apparel_csv_path=os.path.join(os.path.dirname(__file__), 'Apparels_shared.xlsx'), 
                vibe_mapping_path=os.path.join(os.path.dirname(__file__), 'vibe_to_attribute.txt')):
        self.apparel_data = self.load_apparel_data(apparel_csv_path)
        self.vibe_mappings = self.load_vibe_mappings(vibe_mapping_path)
        # Define a list of attributes we might need for filtering and follow-ups, in order of priority
        self.important_attributes = ['category', 'available_sizes', 'price', 'fit', 'fabric', 'sleeve_length', 'occasion']
        # Store conversation state (attributes collected so far)
        self.conversation_attributes = {}
        # Counter for follow-up turns
        self.follow_up_turns_count = 0
        # Store attributes for which the user has indicated no preference
        self.no_preference_attributes = []
        self.last_follow_up_attribute = None # ADDED: To track the last attribute asked in a follow-up

        # Initialize advanced features
        self.vibe_engine = AdvancedVibeEngine()
        self.smart_filtering = SmartFiltering()
        self.conversation_manager = ConversationFlowManager()
        
        # Initialize dynamic keyword mapper
        self.dynamic_mapper = DynamicKeywordMapper(apparel_csv_path)
        print(f"Dynamic keyword mapper initialized with catalog keywords.")
        
        # Configure Google Gemini API
        # Replace with your actual API key or set as an environment variable
        # export GOOGLE_API_KEY='YOUR_API_KEY'
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            print("Please set the GOOGLE_API_KEY environment variable with your Gemini API key.")
            # Exit or handle error appropriately in a real application
            # For this demo, we'll proceed but LLM calls will fail.
            self.model = None
        else:
            genai.configure(api_key=api_key)
            # Choose a suitable model, e.g., 'gemini-pro' or 'gemini-2.0-flash-exp'
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')  # Using Gemini 2.0 Flash
            print("Google Gemini 2.0 Flash model initialized with advanced enhancements.")

        # Initialize recommendation engine after apparel_data is loaded
        self.recommendation_engine = AdvancedRecommendationEngine(self.apparel_data)

    def load_apparel_data(self, data_path):
        # Load the apparel data from the Excel or CSV file
        try:
            if data_path.endswith('.xlsx'):
                df = pd.read_excel(data_path)
            else:
                df = pd.read_csv(data_path, sep='\t')
            print(f"Loaded {len(df)} apparel items from {data_path}.")
            return df
        except FileNotFoundError:
            print(f"Error: Apparel data file not found at {data_path}")
            return pd.DataFrame()
        except Exception as e:
            print(f"Error loading apparel data: {e}")
            return pd.DataFrame()

    def load_vibe_mappings(self, text_path):
        # Load vibe to attribute mappings from the text file
        mappings = {}
        try:
            with open(text_path, 'r') as f:
                content = f.read()
                
                # Use regex to find JSON-like blocks for each vibe
                # Pattern: "vibe_name": { json_object }
                pattern = r'"([^"]+)":\s*(\{[^}]+\})'
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                
                for vibe_name, json_str in matches:
                    try:
                        # Clean up the JSON string and parse it
                        json_str = json_str.replace('\n', ' ').strip()
                        attributes = json.loads(json_str)
                        mappings[vibe_name] = attributes
                    except json.JSONDecodeError as e:
                        print(f"Error parsing JSON for vibe '{vibe_name}': {e}")
                        print(f"JSON string: {json_str}")

            print(f"Loaded {len(mappings)} vibe mappings.")
            return mappings

        except FileNotFoundError:
            print(f"Error: Vibe mapping file not found at {text_path}")
            return {}

    def get_attributes_from_llm(self, query, conversation_attributes, last_asked_attribute=None):
        # Use LLM to extract attributes and identify vibes from the query and determine the next step
        if not self.model:
            print("LLM model not initialized due to missing API key.")
            # Use dynamic mapper fallback
            fallback_attrs = self.dynamic_mapper.fallback_keyword_detection(query) # This method does not exist in dynamic_mapper
            # Corrected to use the actual fallback method from dynamic_mapper or a generic one within agent
            fallback_attrs = self.dynamic_mapper._fallback_keyword_matching(query) if hasattr(self.dynamic_mapper, '_fallback_keyword_matching') else self.fallback_attribute_detection(query)

            no_more_questions = any(phrase in query.lower() for phrase in [
                "no preference", "doesn't matter", "don't care", "anything works", "anything", "any"
            ])
            return fallback_attrs, no_more_questions

        # Try dynamic keyword mapping first, passing the last_asked_attribute
        try:
            # Use the convenience function from dynamic_keyword_mapper.py which now accepts last_asked_attribute
            dynamic_result = get_dynamic_keywords(query, last_asked_attribute=last_asked_attribute)
            if dynamic_result:
                print(f"Dynamic mapper result (with last_asked_attribute='{last_asked_attribute}'): {dynamic_result}")
                
                extracted_attributes = dynamic_result
                
                # Check for no preference indicators
                no_more_questions = any(phrase in query.lower() for phrase in [
                    "no preference", "doesn't matter", "don't care", "anything works", 
                    "anything", "anything is fine", "all good", "i'm flexible"
                ])
                
                return extracted_attributes, no_more_questions
        except Exception as e:
            print(f"Dynamic keyword mapping failed: {e}")
        
        # Fallback if LLM not available or dynamic mapping fails
        # This fallback_attribute_detection is a method of the Agent class
        fallback_attrs = self.fallback_attribute_detection(query)
        no_more_questions = any(phrase in query.lower() for phrase in [
            "no preference", "doesn't matter", "don't care", "anything works", "anything", "any"
        ])
        return fallback_attrs, no_more_questions
        
        # The original LLM prompt logic below is now largely superseded by the call to 
        # get_dynamic_keywords, which itself uses an LLM if available.
        # Keeping the structure for now, but it might need refactoring if we want to use this specific prompt.

        # Get seasonal context for enhanced understanding
        seasonal_context = self.vibe_engine.get_seasonal_attributes()
        style_combinations = self.vibe_engine.analyze_style_combination(query)
        catalog_keywords = self.dynamic_mapper.get_catalog_keywords()

        # Construct an enhanced prompt that guides the LLM
        prompt = f"""You are an advanced vibe-based shopping assistant expert at translating fuzzy human intent into structured apparel attributes.

Your goal is to help users find clothing by understanding their style vibes, seasonal preferences, and extracting relevant filtering attributes.

CONTEXT:
- Current conversation state: {json.dumps(conversation_attributes, indent=2)}
- Available catalog keywords: {json.dumps({k: list(v)[:10] for k, v in catalog_keywords.items()}, indent=2)}
- Current season context: {json.dumps(seasonal_context, indent=2)}
- Detected style combinations: {json.dumps(style_combinations, indent=2)}
- Available attributes: category, available_sizes, price, fit, fabric, sleeve_length, occasion, color_or_print, neckline, length, pant_type

ENHANCED CAPABILITIES:
- Seasonal awareness: Consider current season for fabric, color, and style suggestions
- Style combination understanding: Recognize complex style descriptors like "boho chic", "minimalist", etc.
- Advanced color matching: Understand color families and complementary colors
- Price intelligence: Interpret budget hints like "affordable", "luxury", "splurge"
- Trend awareness: Consider trending styles and current fashion

TASK:
Analyze the user's current query and extract:
1. Explicit attributes mentioned (size, price, category, etc.)
2. Vibe terms that can be mapped to attributes using the vibe mappings
3. Style combinations and seasonal preferences
4. Whether user wants no more follow-up questions

IMPORTANT RULES:
- For sizes: Use exact format like "S", "M", "L", "XL" 
- For price: Extract as numeric value (e.g., 100 for $100) or price hints
- For categories: Use "top", "dress", "pants", "skirt"
- For specific pant types: Use pant_type attribute (e.g., "cargo", "denim")
- If user mentions "cargo" or "cargos", set category to "pants" and pant_type to "cargo"
- SEASONAL AWARENESS: If user mentions "winter", "summer", "spring", "fall" or seasonal phrases like "winter options", extract seasonal preferences:
  * "winter" → fabric: ["Wool", "Fleece", "Cotton"], sleeve_length: ["Long sleeves", "Full sleeves"]
  * "summer" → fabric: ["Linen", "Cotton"], sleeve_length: ["Short sleeves", "Sleeveless"], category: "dress"
  * "spring" → fabric: ["Cotton", "Linen"], sleeve_length: ["Short sleeves", "Long sleeves"]
  * "fall/autumn" → fabric: ["Cotton", "Wool"], sleeve_length: ["Long sleeves", "Full sleeves"]
- Consider seasonal appropriateness in recommendations
- The system has comprehensive keyword mapping and advanced filtering
- Focus on extracting explicit attributes and clear vibes mentioned by the user
- If user says "no preference", "any", "doesn't matter", "don't care", "anything works" for ANY attribute, set no_more_questions to true
- Detect if user wants to skip questions (e.g., "just show me", "no more questions")
- If user is answering a specific follow-up question but indicates no preference, treat this as wanting to proceed

ADVANCED KEYWORD AWARENESS:
The system automatically maps these types of keywords with enhanced understanding:
- Style vibes: comfy, fitted, cozy, relaxed, oversized, cropped, etc.
- Complex styles: boho chic, minimalist, romantic, edgy, preppy
- Occasions: work, party, casual, formal, date night, brunch, etc.
- Fabrics: cotton, silk, linen, satin, wool, cashmere, etc.
- Colors: Full color families, seasonal colors, trending colors
- Fit types: body hugging, tailored, relaxed, oversized, fitted
- Categories and specific items: cargo, jeans, dress, blazer, etc.
- Price hints: budget, affordable, luxury, premium, splurge
- Seasonal terms: spring fresh, summer breezy, fall cozy, winter warm

Focus on what the user explicitly states while considering seasonal and trend context.

Respond ONLY with valid JSON:
{{
   "extracted_attributes": {{}},
   "identified_vibes": [],
   "no_more_questions": false,
   "seasonal_relevance": "",
   "style_confidence": 0.0
}}

User Query: {query}

JSON Output:"""

        try:
            response = self.model.generate_content(prompt)
            # Assuming the LLM response content is the JSON string
            response_text = response.text.strip()
            # Clean the response text to ensure it's valid JSON
            # Sometimes LLMs might add markdown or extra text
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            llm_output = json.loads(response_text)
            extracted_attributes = llm_output.get('extracted_attributes', {})
            identified_vibes = llm_output.get('identified_vibes', [])
            no_more_questions = llm_output.get('no_more_questions', False)
            
            print(f"LLM Extracted Attributes: {extracted_attributes}")
            print(f"LLM Identified Vibes: {identified_vibes}")
            print(f"LLM No More Questions Flag: {no_more_questions}")

            # Infer attributes from identified vibes using our loaded mappings
            inferred_attributes = self.infer_attributes_from_vibes(identified_vibes)
            print(f"Inferred Attributes from Vibes: {inferred_attributes}")

            # Combine extracted and inferred attributes. Inferred attributes are a lower priority
            # unless they add a new attribute not already explicitly mentioned or inferred.
            # A simple merge for now, explicit takes precedence:
            all_identified_attributes = {**inferred_attributes, **extracted_attributes}

            # Return identified attributes and the no_more_questions flag
            return all_identified_attributes, no_more_questions

        except Exception as e:
            print(f"Error calling LLM or parsing response: {e}")
            # Fallback: Use dynamic keyword mapper
            fallback_attrs = self.dynamic_mapper.fallback_keyword_detection(query)
            no_more_questions = any(phrase in query.lower() for phrase in [
                "no preference", "doesn't matter", "don't care", "anything works", "anything", "any"
            ])
            return fallback_attrs, no_more_questions

    def process_query(self, query):
        # This method now handles processing both initial queries and follow-up responses
        print(f"Processing query: {query}")
        
        # Check if user wants to reset the conversation state (only if they explicitly mention "reset")
        query_lower = query.lower()
        if self.should_reset_conversation(query):
            self.reset_conversation_state()
            # Return a special response to confirm the reset
            return {
                "response_type": "message", 
                "message": "I've reset our conversation. What are you looking for today?",
                "attributes": {}
            }
        
        # --- New Architecture Steps ---

        # Step 1: Initial Broad Keyword Retrieval (can be skipped if conversation_attributes is not empty)
        initial_results_df = None
        query_lower = query.lower()
        # If the conversation state is empty, perform a new initial keyword search
        if not self.conversation_attributes:
            keywords = query_lower.split() # Potential issue: split creates too many keywords
            # Refined keyword extraction: prioritize multi-word phrases from query if they match known attributes/vibes
            # For simplicity, we'll stick to the split for now but this is an area for improvement.
            keyword_mask = None
            for keyword in keywords:
                name_match = self.apparel_data['name'].astype(str).str.contains(keyword, case=False, na=False)
                category_match = self.apparel_data['category'].astype(str).str.contains(keyword, case=False, na=False)
                if keyword_mask is None:
                    keyword_mask = name_match | category_match
                else:
                    keyword_mask = keyword_mask | name_match | category_match
            if keyword_mask is not None and not keyword_mask.empty:
                initial_results_df = self.apparel_data[keyword_mask].copy()
            else:
                # If initial keyword search yields no results, work with the full catalog initially
                initial_results_df = self.apparel_data.copy()
                print(f"Initial keyword search for keywords {keywords} in 'name' and 'category' returned 0 items. Using full catalog for initial filtering.")

            print(f"Initial keyword search for keywords {keywords} in 'name' and 'category' returned {len(initial_results_df)} items.")
        else:
            # If conversation_attributes exist, we are refining a previous search.
            # In this simplified version, we will re-filter the entire catalog based on the updated attributes.
            # A more advanced version could refine the *previous* initial_results_df subset.
            print("Continuing previous search, skipping initial keyword retrieval.")
            # Use the full apparel data as the base for filtering in subsequent turns
            initial_results_df = self.apparel_data.copy()

        # Step 2: LLM-driven Attribute Generation & Inference
        # Pass the current query, conversation state, and last follow-up attribute to the LLM/mapper
        identified_attributes, no_more_questions = self.get_attributes_from_llm(query, self.conversation_attributes, self.last_follow_up_attribute)

        print(f"Identified attributes from LLM (considering last_follow_up: {self.last_follow_up_attribute}): {identified_attributes}")

        # Clear last_follow_up_attribute after it has been used
        # self.last_follow_up_attribute = None 
        # ^ Let's move this to after we are sure we are not asking the same question again.

        # Step 3: Merge newly identified attributes with the existing conversation state
        # Create a copy to avoid modifying the dictionary during iteration
        updated_state = self.conversation_attributes.copy()
        
        # Check if the user wants to show all items (clear all filters)
        if identified_attributes.get('show_all', False):
            print("User requested to show all items. Clearing all conversation state.")
            updated_state = {}
            self.conversation_attributes = {}
            self.no_preference_attributes = []
            # Return all items immediately
            all_items = self.apparel_data.head(20)[['id', 'name', 'price']].to_dict(orient='records')
            justification = "Here's our complete catalog of items for you to browse!"
            return {"response_type": "recommendation", "recommendations": all_items, "justification": justification, "attributes": {}}
        
        if identified_attributes:
            for attr, value in identified_attributes.items():
                if attr == 'show_all':  # Skip the show_all flag
                    continue
                if value is None:
                    # If LLM explicitly set to None (no preference), remove from conversation attributes if present
                    if attr in updated_state:
                        del updated_state[attr]
                else:
                    # If value is not None, update or add the attribute to the state
                    updated_state[attr] = value
        
        # Special case: If pant_type is specified but category is not, infer category as pants
        if 'pant_type' in updated_state and 'category' not in updated_state:
            updated_state['category'] = 'pants'
            print("Inferred category as 'pants' based on pant_type specification")

        # Update the list of attributes for which the user has indicated no preference
        if identified_attributes:
            for attr, value in identified_attributes.items():
                if attr == 'show_all':  # Skip the show_all flag
                    continue
                if value is None and attr not in self.no_preference_attributes:
                    self.no_preference_attributes.append(attr)
                    print(f"Added '{attr}' to no preference list")

        self.conversation_attributes = updated_state

        print(f"Current Conversation Attributes after merge: {self.conversation_attributes}")
        print(f"No preference attributes: {self.no_preference_attributes}")

        # Step 4: Refined Filtering and Scoring
        # Filter the apparel data based on the current conversation attributes.
        # Use the initial_results_df (which might be the full catalog or keyword subset)
        filtered_df = self.filter_apparel(self.conversation_attributes, initial_results_df)

        # For now, simple filtering. Scoring can be added later based on how well items match all attributes.
        # A simple scoring could be the number of matching attributes.
        # For this iteration, let's just use the filtered_df directly.

        print(f"Filtered down to {len(filtered_df)} items after applying conversation attributes.")

        # Step 5: Enhanced Agent Decision with Smart Filtering
        follow_up_questions = []
        next_step = 'recommendation' # Default to recommendation

        # Enhanced criteria for asking follow-up questions:
        # - If no results, try relaxing some constraints first
        # - Only ask follow-up if we really need more info after trying flexible filtering
        
        # If we got 0 results, try relaxing constraints intelligently
        if len(filtered_df) == 0 and self.conversation_attributes:
            print("No results found with current filters. Attempting intelligent constraint relaxation...")
            filtered_df = self.apply_flexible_filtering(self.conversation_attributes, initial_results_df)
            print(f"After flexible filtering: {len(filtered_df)} items found")

        # Determine the most important missing attribute to ask about if needed
        missing_attributes = []
        print(f"DEBUG: Follow-up decision - filtered_df size: {len(filtered_df)}, follow_up_count: {self.follow_up_turns_count}, no_more_questions: {no_more_questions}")
        
        # IMPORTANT: Only consider asking follow-up questions if we haven't reached the limit of 2 questions
        # This is a hard limit to prevent repetitive questioning
        if self.follow_up_turns_count >= 2:
            print("DEBUG: Already asked 2 follow-up questions - no more follow-ups allowed")
            missing_attributes = []  # Force empty to skip follow-ups
        elif (len(filtered_df) > 15 or len(filtered_df) == 0) and not no_more_questions:
            # Only look for missing attributes if we're under the limit and conditions warrant follow-up questions
            for attr in self.important_attributes:
                # Ask about attributes that are NOT in the current conversation state AND have not been marked as no preference
                # AND was not the *immediately* preceding follow-up question (to avoid re-asking)
                if attr not in self.conversation_attributes and attr not in self.no_preference_attributes and attr != self.last_follow_up_attribute:
                    missing_attributes.append(attr)
        print(f"DEBUG: Missing attributes to ask about: {missing_attributes}")

        if missing_attributes:
            next_step = 'follow_up'
            # Generate enhanced contextual questions
            # Pass the *chosen* missing attribute to generate_smart_follow_up_questions
            chosen_attribute_to_ask = missing_attributes[0] # Take the first one for now
            follow_up_questions = self.generate_smart_follow_up_questions([chosen_attribute_to_ask], query) # Pass as a list
            
            # Fallback to basic questions if enhanced generation fails
            if not follow_up_questions:
                # missing_attribute = missing_attributes[0] # Already defined as chosen_attribute_to_ask
                if chosen_attribute_to_ask == 'category':
                    follow_up_questions.append("Do you have a preference between dresses, tops & skirts, or pants?")
                elif chosen_attribute_to_ask == 'available_sizes':
                    follow_up_questions.append("What size are you looking for? (e.g., S, M, L, XL)")
                elif chosen_attribute_to_ask == 'price':
                    follow_up_questions.append("What's your budget range?")
                elif chosen_attribute_to_ask == 'fit':
                    follow_up_questions.append("What kind of fit do you prefer (e.g., relaxed, tailored, body hugging)?")
                elif chosen_attribute_to_ask == 'fabric':
                    follow_up_questions.append("Any preference on fabric (e.g., cotton, silk, linen)?")
                elif chosen_attribute_to_ask == 'sleeve_length':
                    follow_up_questions.append("What sleeve length are you looking for?")
                elif chosen_attribute_to_ask == 'occasion':
                    follow_up_questions.append("What's the occasion?")

            # Ensure only one follow-up question is generated per turn
            if follow_up_questions:
                 follow_up_questions = [follow_up_questions[0]]
                 self.last_follow_up_attribute = chosen_attribute_to_ask # SET the last asked attribute
            else:
                # If a missing attribute was found but no specific question was generated, default to recommendation
                next_step = 'recommendation'
                self.last_follow_up_attribute = None # Clear if no question is asked
        else:
             self.last_follow_up_attribute = None # Clear if no missing attributes to ask about


        # Step 6: Prepare and Return Response
        if next_step == 'follow_up' and follow_up_questions:
            print(f"DEBUG: Entering agent-driven follow-up flow. Follow-up count: {self.follow_up_turns_count}")
            print(f"DEBUG: Agent Follow-up questions: {follow_up_questions}")
            print(f"DEBUG: Last asked attribute set to: {self.last_follow_up_attribute}")
            self.follow_up_turns_count += 1
            return {"response_type": "follow_up", "questions": follow_up_questions, "attributes": self.conversation_attributes}
        else:
            print("\nProceeding to enhanced recommendations...")
            self.last_follow_up_attribute = None # Ensure it's cleared if we are recommending
            
            # Apply additional advanced filtering if needed
            filtered_df = apply_advanced_filtering(filtered_df, query, self.smart_filtering)
            print(f"DEBUG: After advanced filtering: {len(filtered_df)} items remain")
            
            # Generate enhanced recommendations with advanced features
            enhanced_recommendations = generate_enhanced_recommendations(self, filtered_df, self.conversation_attributes)
            print(f"DEBUG: Enhanced recommendations returned: {len(enhanced_recommendations)} items")
            if enhanced_recommendations:
                print(f"DEBUG: First recommendation sample: {enhanced_recommendations[0].get('name', 'Unknown')}, ${enhanced_recommendations[0].get('price', 0)}")
            
            # Generate enhanced justification
            justification = self.generate_enhanced_justification(self.conversation_attributes, query)
            
            # Prepare recommendations with additional features
            recommended_items = []
            for item in enhanced_recommendations[:7]:  # Limit to 7 items for better UX
                # Convert item to a clean dictionary with only needed attributes
                rec_item = {
                    'id': item.get('id', '') if isinstance(item, dict) else str(getattr(item, 'id', '')),
                    'name': item.get('name', '') if isinstance(item, dict) else getattr(item, 'name', ''),
                    'price': item.get('price', 0) if isinstance(item, dict) else getattr(item, 'price', 0),
                    'is_trending': item.get('is_trending', False) if isinstance(item, dict) else getattr(item, 'is_trending', False)
                }
                
                # Add outfit suggestions if available
                if isinstance(item, dict) and 'outfit_suggestions' in item and item['outfit_suggestions']:
                    rec_item['outfit_suggestions'] = item['outfit_suggestions'][:2]  # Limit suggestions
                
                recommended_items.append(rec_item)
            
            # Remember user preferences for future sessions
            if hasattr(self, 'conversation_manager'):
                for attr, value in self.conversation_attributes.items():
                    self.conversation_manager.remember_preference('default_user', attr, value)
            
            # Reset follow-up counter but keep conversation state for refinements
            self.follow_up_turns_count = 0
            print("Follow-up counter reset. Conversation state preserved for refinements.")

            return {
                "response_type": "recommendation", 
                "recommendations": recommended_items, 
                "justification": justification, 
                "attributes": {},
                "enhanced_features": True
            }

    # The old extract_attributes_from_query and identify_vibes_from_query are now less critical but can be kept as fallback or for simpler cases.
    # However, for this LLM-based approach, we will primarily rely on the LLM output.
    # Keeping them here for reference for now, but they are not directly used in the new process_query flow.

    def extract_attributes_from_query_keyword(self, query):
        # More sophisticated keyword extraction (deprecated in favor of LLM for main flow)
        extracted = {}
        query_lower = query.lower()

        # Category
        if "dress" in query_lower or "dresses" in query_lower:
            extracted["category"] = ["dress"]
        if "top" in query_lower or "tops" in query_lower:
            if "category" in extracted: # Handle multiple categories requested
                 extracted["category"].append("top")
            else:
                 extracted["category"] = ["top"]
        if "skirt" in query_lower or "skirts" in query_lower:
             if "category" in extracted:
                  extracted["category"].append("skirt")
             else:
                  extracted["category"] = ["skirt"]
        if "jean" in query_lower or "jeans" in query_lower:
             if "category" in extracted:
                  extracted["category"].append("pants") # Assuming jeans fall under pants category in data
             else:
                  extracted["category"] = ["pants"]
        if "pant" in query_lower or "pants" in query_lower:
             if "category" in extracted:
                  extracted["category"].append("pants")
             else:
                  extracted["category"] = ["pants"]

        # Size (basic extraction)
        size_match = re.search(r'size\s+([xsmlXSML]+)', query_lower)
        if size_match:
            extracted["size"] = [size_match.group(1).upper()]

        # Budget (basic extraction - handling ranges like 5-6k and single values)
        price_match_range = re.search(r'between\s*\$?(\d+)[kK]?\s*-\s*\$?(\d+)[kK]?', query_lower)
        price_match_under = re.search(r'under\s*\$?(\d+)[kK]?', query_lower)
        price_match_above = re.search(r'above\s*\$?(\d+)[kK]?', query_lower)
        price_match_exactly = re.search(r'\$(\d+)[kK]?', query_lower)
        price_match_budget = re.search(r'budget\s+is.*?(\d+)\$?[kK]?', query_lower)  # "budget is like 100$"
        price_match_trailing = re.search(r'(\d+)\$[kK]?', query_lower)  # "100$" format

        if price_match_range:
            min_price = int(price_match_range.group(1).replace('k', '000'))
            max_price = int(price_match_range.group(2).replace('k', '000'))
            extracted["price_min"] = min_price
            extracted["price_max"] = max_price
        elif price_match_under:
            max_price = int(price_match_under.group(1).replace('k', '000'))
            extracted["price_max"] = max_price
        elif price_match_above:
             min_price = int(price_match_above.group(1).replace('k', '000'))
             extracted["price_min"] = min_price
        elif price_match_exactly:
             exact_price = int(price_match_exactly.group(1).replace('k', '000'))
             extracted["price_min"] = exact_price
             extracted["price_max"] = exact_price
        elif price_match_budget:
             budget_price = int(price_match_budget.group(1).replace('k', '000'))
             extracted["price_max"] = budget_price
        elif price_match_trailing:
             trailing_price = int(price_match_trailing.group(1).replace('k', '000'))
             extracted["price_max"] = trailing_price

        # Sleeve length (basic extraction)
        if "sleeveless" in query_lower:
            extracted["sleeve_length"] = "Sleeveless"
        if "short sleeve" in query_lower or "short-sleeve" in query_lower:
             extracted["sleeve_length"] = "Short sleeves"
        if "long sleeve" in query_lower or "long-sleeve" in query_lower:
             extracted["sleeve_length"] = "Long sleeves"
        # Add more sleeve lengths as needed

        # Fit (basic extraction)
        if "relaxed fit" in query_lower or "flowy" in query_lower:
            extracted["fit"] = "Relaxed"
        if "bodycon" in query_lower or "body hugging" in query_lower:
            extracted["fit"] = "Body hugging"
        if "tailored fit" in query_lower:
            extracted["fit"] = "Tailored"
        # Add more fits as needed

        # Fabric (basic extraction)
        if "linen" in query_lower: extracted["fabric"] = ["Linen"]
        if "cotton" in query_lower: 
             if "fabric" in extracted:
                  extracted["fabric"].append("Cotton")
             else:
                  extracted["fabric"] = ["Cotton"]
        # Add more fabrics and handle multiple fabrics
        fabric_matches = re.findall(r'(linen|cotton|silk|velvet|satin)', query_lower)
        if fabric_matches:
             if "fabric" not in extracted:
                  extracted["fabric"] = []
             for fabric in fabric_matches:
                 capitalized_fabric = fabric.capitalize()
                 if capitalized_fabric not in extracted["fabric"]:
                      extracted["fabric"].append(capitalized_fabric)

        # Occasion (basic extraction)
        if "brunch" in query_lower: extracted["occasion"] = "Everyday" # Mapping brunch to Everyday occasion
        if "party" in query_lower: extracted["occasion"] = "Party"
        if "vacation" in query_lower or "vacay" in query_lower: extracted["occasion"] = "Vacation"
        if "work" in query_lower or "office" in query_lower: extracted["occasion"] = "Work"

        return extracted

    def identify_vibes_from_query_keyword(self, query):
        # Identify potential vibe terms from the query (deprecated in favor of LLM for main flow)
        identified = []
        query_lower = query.lower()
        # This is still basic, could use embedding/LLM for better vibe detection
        # Check for vibes directly present in the keys of vibe_mappings
        for vibe_key in self.vibe_mappings.keys():
             if vibe_key in query_lower:
                  identified.append(vibe_key)
        
        # Also check for individual vibe words if not part of a specific phrase
        individual_vibe_words = ["cute", "elevated", "comfy", "polish", "flowy", "glamorous", "beachy", "retro"]
        for word in individual_vibe_words:
             if word in query_lower and word not in ''.join(identified): # Avoid adding individual word if part of a phrase already found
                  identified.append(word)

        # Simple filtering of duplicates
        identified = list(dict.fromkeys(identified))

        print(f"Identified Vibes (Keyword): {identified}")
        return identified
        

    def infer_attributes_from_vibes(self, vibes):
        # Infer attributes based on identified vibes using the mappings
        inferred = {}
        for vibe in vibes:
            if vibe in self.vibe_mappings:
                for attr, value in self.vibe_mappings[vibe].items():
                     # Simple merge, overwrite if conflict. Could be more sophisticated.
                     # For list values (like fabric), extend the list
                     if isinstance(value, list) and attr in inferred and isinstance(inferred[attr], list):
                          inferred[attr].extend(value)
                          # Remove duplicates
                          inferred[attr] = list(dict.fromkeys(inferred[attr]))
                     else:
                          inferred[attr] = value
        return inferred

    def determine_follow_up_questions(self, current_attributes):
        # Determine which follow-up questions to ask (max 2)
        questions = []
        asked_count = 0
        
        print(f"Determining follow-up questions based on attributes: {current_attributes}")

        # Prioritized list of attributes to ask about if missing
        attributes_to_ask = ['category', 'size', 'price_max', 'fit', 'fabric', 'sleeve_length', 'occasion']
        
        # Check for combined size and budget question first
        if asked_count < 2 and 'size' not in current_attributes and ('price_max' not in current_attributes and 'price_min' not in current_attributes):
             questions.append("Any must-haves like size or budget range to keep in mind?")
             asked_count += 1
        
        # Iterate through other important attributes and ask about missing ones individually if needed
        for attr in attributes_to_ask:
            # Skip size and price_max if the combined question was asked
            if attr in ['size', 'price_max'] and len(questions) > 0 and questions[0] == "Any must-haves like size or budget range to keep in mind?":
                 continue
            
            # Skip asking about price_max individually if price_min is present and vice versa (handled below)
            if attr == 'price_max' and 'price_min' in current_attributes and 'price_max' not in current_attributes:
                 if asked_count < 2:
                     questions.append(f"Okay, you mentioned a price around ${current_attributes['price_min']}. Do you have a maximum budget in mind?")
                     asked_count += 1
                 continue
            elif attr == 'price_max' and 'price_max' in current_attributes and 'price_min' not in current_attributes: # Less likely scenario
                 if asked_count < 2:
                      questions.append(f"Okay, you mentioned a maximum price of ${current_attributes['price_max']}. Do you have a minimum budget in mind?")
                      asked_count += 1
                 continue

            # Ask about other missing attributes individually if needed
            if asked_count < 2 and attr not in current_attributes and attr != 'price_min': # Don't ask about price_min individually unless handling a max price response
                if attr == 'category':
                    questions.append("Do you have a preference between dresses, tops & skirts, or something more casual like jeans?")
                elif attr == 'size':
                     # Check if size is missing or explicitly indicated as 'any'
                     size_value = current_attributes.get('size')
                     if not size_value or (isinstance(size_value, str) and size_value.lower() == 'any') or (isinstance(size_value, list) and len(size_value) == 0) or (isinstance(size_value, list) and size_value[0].lower() == 'any'):
                         questions.append("What size are you looking for?")
                # Price_max is handled in the combined question or specific min/max follow-up
                elif attr == 'fit':
                     questions.append("What kind of fit do you prefer (e.g., relaxed, tailored, bodycon)?")
                elif attr == 'fabric':
                     questions.append("Any preference on fabric (e.g., cotton, silk, linen)?")
                elif attr == 'sleeve_length':
                     questions.append("What sleeve length are you looking for?")
                elif attr == 'occasion':
                     questions.append("What's the occasion?")
                # Add more attribute-specific questions here
                asked_count += 1

        return questions

    def filter_apparel(self, attributes, df_to_filter=None):
        # Enhanced filtering with advanced features while preserving existing logic
        filtered_df = df_to_filter.copy() if df_to_filter is not None else self.apparel_data.copy()
        print(f"Initial items before filtering in filter_apparel: {len(filtered_df)}")
        print(f"DEBUG: All attributes being used for filtering: {attributes}")
        
        # Apply seasonal intelligence if no specific attributes are provided
        if not attributes:
            seasonal_attrs = self.vibe_engine.get_seasonal_attributes()
            print(f"No attributes provided, applying seasonal intelligence: {seasonal_attrs}")
        
        for attr, value in attributes.items():
            # Skip filtering for attributes where the user indicated no preference (value is None)
            if value is None:
                print(f"Skipping filter for '{attr}' as value is None.")
                continue

            print(f"Applying filter: {attr} = {value}")
            
            # Handle size filtering - the data has 'available_sizes' column
            if attr == 'available_sizes' or attr == 'size':
                if 'available_sizes' in filtered_df.columns:
                    if isinstance(value, list):
                        # Check if any of the requested sizes are available
                        size_pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                        filtered_df = filtered_df[filtered_df['available_sizes'].astype(str).str.contains(size_pattern, case=False, na=False)]
                    else:
                        filtered_df = filtered_df[filtered_df['available_sizes'].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
                    print(f"Items after size filter: {len(filtered_df)}")
                continue
                
            # Enhanced price filtering with intelligence
            if attr == 'price' or attr == 'price_max':
                if 'price' in filtered_df.columns:
                    # Debug print to see what's happening with price filtering
                    print(f"DEBUG: Filtering price with value {value}")
                    print(f"DEBUG: Price column sample before filtering: {filtered_df['price'].head().tolist()}")
                    
                    # Filter for items at or under the specified price
                    filtered_df = filtered_df[pd.to_numeric(filtered_df['price'], errors='coerce') <= value]
                    print(f"Items after price filter: {len(filtered_df)}")
                    
                    # Debug print to see results after price filtering
                    if len(filtered_df) > 0:
                        print(f"DEBUG: Price column sample after filtering: {filtered_df['price'].head().tolist()}")
                    else:
                        print(f"DEBUG: No items match the price filter of {value}")
                continue
            
            # Enhanced color filtering with palette matching
            if attr == 'color_or_print' or attr == 'color':
                if 'color_or_print' in filtered_df.columns:
                    # Use smart color palette matching
                    filtered_df = self.smart_filtering.match_color_palette(str(value), filtered_df)
                    print(f"Items after enhanced color filter: {len(filtered_df)}")
                continue
            
            # Enhanced fabric filtering with fuzzy matching
            elif attr == 'fabric':
                if 'fabric' in filtered_df.columns:
                    if isinstance(value, list):
                        pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                        filtered_df = filtered_df[filtered_df['fabric'].astype(str).str.contains(pattern, case=False, na=False)]
                    else:
                        # Try fuzzy matching for fabric
                        fabric_matches = self.smart_filtering.fuzzy_match_attributes(str(value), filtered_df['fabric'].unique())
                        if fabric_matches and fabric_matches[0][1] > 0.7:  # High confidence threshold
                            best_match = fabric_matches[0][0]
                            filtered_df = filtered_df[filtered_df['fabric'].astype(str).str.contains(re.escape(best_match), case=False, na=False)]
                            print(f"Used fuzzy matching: '{value}' -> '{best_match}'")
                        else:
                            # Fallback to exact matching
                            filtered_df = filtered_df[filtered_df['fabric'].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
                    print(f"Items after enhanced fabric filter: {len(filtered_df)}")
                continue
            
            # Special handling for pant_type - search in name column when category is pants
            elif attr == 'pant_type':
                if isinstance(value, list):
                    pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                    filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains(pattern, case=False, na=False)]
                else:
                    # Handle specific pant types - make search case-insensitive
                    if value.lower() in ["cargo", "cargos"]:
                        filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains("cargo", case=False, na=False)]
                    elif value.lower() in ["denim", "jean", "jeans"]:
                        filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains("denim|jean", case=False, na=False)]
                    else:
                        filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
                print(f"Items after {attr} filter (searching name column): {len(filtered_df)}")
                print(f"DEBUG: Searching for '{value}' in name column, found {len(filtered_df)} items")
            
            # Handle other attributes that exist in the dataframe
            elif attr in filtered_df.columns:
                if isinstance(value, list):
                    # Handle attributes with multiple possible values (e.g., category, fabric)
                    pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                    if attr == 'category':
                        # Special handling for category - trim whitespace in data for comparison
                        filtered_df = filtered_df[filtered_df[attr].astype(str).str.strip().str.contains(pattern, case=False, na=False)]
                    else:
                        filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(pattern, case=False, na=False)]
                    print(f"Items after {attr} filter: {len(filtered_df)}")
                else:
                    # Handle attributes with single values (e.g., fit, sleeve_length, occasion)
                    if attr == 'category':
                        # Special handling for category - trim whitespace in data for comparison
                        filtered_df = filtered_df[filtered_df[attr].astype(str).str.strip().str.contains(re.escape(str(value).strip()), case=False, na=False)]
                    else:
                        filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
                    print(f"Items after {attr} filter: {len(filtered_df)}")
        
        print(f"Final items after filtering in filter_apparel: {len(filtered_df)}")
        return filtered_df

    def generate_justification(self, attributes):
        # Generate a justification based on the attributes used for filtering
        justification_parts = []
        
        if "occasion" in attributes:
            justification_parts.append(f"your {attributes['occasion']} occasion vibe")
        
        if "fit" in attributes:
            justification_parts.append(f"{attributes['fit']} fit")
            
        if "fabric" in attributes:
             if isinstance(attributes['fabric'], list):
                  justification_parts.append(f"breathable fabrics like {' and '.join(attributes['fabric'])}")
             else:
                  justification_parts.append(f"{attributes['fabric']} fabric")
                  
        if "sleeve_length" in attributes:
             justification_parts.append(f"{attributes['sleeve_length']}")
             
        if "category" in attributes:
             if isinstance(attributes['category'], list):
                  justification_parts.append(f"{ ' and '.join(attributes['category'])}")
             else:
                  justification_parts.append(f"{attributes['category']}")
        
        price_parts = []
        if "price" in attributes:
             price_parts.append(f"under ${attributes['price']}")
        elif "price_min" in attributes and "price_max" in attributes:
             if attributes['price_min'] == attributes['price_max']:
                  price_parts.append(f"around ${attributes['price_min']}")
             else:
                  price_parts.append(f"between ${attributes['price_min']} and ${attributes['price_max']}")
        elif "price_min" in attributes:
             price_parts.append(f"above ${attributes['price_min']}")
        elif "price_max" in attributes:
             price_parts.append(f"under ${attributes['price_max']}")

        if price_parts:
             justification_parts.append(" with price " + " and ".join(price_parts))
             
        if "available_sizes" in attributes or "size" in attributes:
             size_attr = attributes.get('available_sizes') or attributes.get('size')
             if isinstance(size_attr, list):
                  justification_parts.append(f"that match your size(s) {' and '.join(size_attr)}")
             else:
                  justification_parts.append(f"that match your size {size_attr}")

        if not justification_parts:
            return "Based on your vibe preferences, I have selected these items for you."
            
        # Construct the final justification sentence
        return "Based on " + ", ".join(justification_parts) + ", I found these perfect matches for you."

    def fallback_attribute_detection(self, query):
        """Enhanced fallback method using comprehensive keyword mapping system"""
        detected = {}
        query_lower = query.lower().strip()
        
        # Detect "show all" commands
        if any(phrase in query_lower for phrase in ["show me all", "show all", "whole catalog", "entire catalog", "everything", "all items"]):
            detected['show_all'] = True
            return detected
        
        # Enhanced "no preference" responses handling - check first
        if any(phrase in query_lower for phrase in [
            "no preference", "doesn't matter", "don't care", "any", "any size", 
            "doesn't matter to me", "don't mind", "whatever", "not picky",
            "no specific preference", "not particular", "all sizes",
            "anything works", "anything", "anything is fine", "all good",
            "no preference really", "either way", "i'm flexible"
        ]):
            detected['no_more_questions'] = True
            # If it's a response to a size question, explicitly mark size as no preference
            if any(word in query_lower for word in ["size", "any size", "all sizes"]):
                detected["available_sizes"] = None
            return detected
            
        elif query_lower.strip() in ["no", "nope", "none", "n/a", "na"]:
            # Simple "no" could be a response to a category or other question
            detected['no_more_questions'] = True
            return detected
        
        # Handle "no budget" specifically  
        if any(phrase in query_lower for phrase in ["no budget", "budget doesn't matter", "any budget", "any price"]):
            detected["price"] = None  # Remove price constraint
            detected['no_more_questions'] = True
            return detected
        
        # Use the comprehensive keyword mapping system
        keyword_attributes = search_keywords(query_lower)
        if keyword_attributes:
            detected.update(keyword_attributes)
            print(f"[DEBUG] Keyword mapping detected: {keyword_attributes}")
        
        # Additional seasonal detection for fallback (in case keyword mapping misses compound phrases)
        if "winter options" in query_lower or "winter vibes" in query_lower:
            detected.update({"fabric": ["Wool", "Fleece", "Cotton"], "sleeve_length": ["Long sleeves", "Full sleeves"]})
        elif "summer options" in query_lower or "summer vibes" in query_lower:
            detected.update({"fabric": ["Linen", "Cotton"], "sleeve_length": ["Short sleeves", "Sleeveless"]})
        elif "spring options" in query_lower:
            detected.update({"fabric": ["Cotton", "Linen"], "sleeve_length": ["Short sleeves", "Long sleeves"]})
        elif "fall options" in query_lower or "autumn options" in query_lower:
            detected.update({"fabric": ["Cotton", "Wool"], "sleeve_length": ["Long sleeves", "Full sleeves"]})
        
        # Manual size detection (more specific patterns)
        size_patterns = [r'\b([smlxSMLX]+)\b', r'size\s+([smlxSMLX]+)', r'([smlxSMLX]+)\s+size']
        for pattern in size_patterns:
            size_match = re.search(pattern, query_lower)
            if size_match:
                detected["available_sizes"] = size_match.group(1).upper()
                break
        
        # Price detection
        price_patterns = [
            (r'under\s+\$?(\d+)', 'max'),
            (r'below\s+\$?(\d+)', 'max'),
            (r'above\s+\$?(\d+)', 'min'),
            (r'over\s+\$?(\d+)', 'min'),
            (r'\$(\d+)', 'exact')
        ]
        
        for pattern, price_type in price_patterns:
            price_match = re.search(pattern, query_lower)
            if price_match:
                price_val = int(price_match.group(1))
                if price_type == 'max':
                    detected["price"] = price_val
                elif price_type == 'exact':
                    detected["price"] = price_val
                break
        
        return detected

    def generate_enhanced_justification(self, attributes, original_query):
        """Generate enhanced justification with seasonal and style context"""
        justification_parts = []
        
        # Get seasonal context
        seasonal_attrs = self.vibe_engine.get_seasonal_attributes()
        current_season = self.get_current_season()
        
        # Analyze style combinations from original query
        style_combinations = self.vibe_engine.analyze_style_combination(original_query)
        
        # Build justification with enhanced context
        if style_combinations:
            style_names = list(style_combinations.keys())
            if len(style_names) == 1:
                justification_parts.append(f"your {style_names[0]} style preference")
            else:
                justification_parts.append(f"your {' and '.join(style_names)} aesthetic")
        
        if "occasion" in attributes:
            justification_parts.append(f"perfect for {attributes['occasion']} occasions")
        
        if "fit" in attributes:
            justification_parts.append(f"featuring {attributes['fit']} fit")
            
        if "fabric" in attributes:
            if isinstance(attributes['fabric'], list):
                justification_parts.append(f"made with {' and '.join(attributes['fabric'])} fabrics")
            else:
                justification_parts.append(f"crafted in {attributes['fabric']}")
                
        if "color_or_print" in attributes:
            justification_parts.append(f"in your preferred {attributes['color_or_print']} tones")
        
        # Add seasonal context if relevant - detect user's seasonal preference from query
        user_seasonal_preference = None
        query_lower = original_query.lower()
        if any(season in query_lower for season in ["winter", "winter options", "winter vibes", "winter weather"]):
            user_seasonal_preference = "winter"
        elif any(season in query_lower for season in ["summer", "summer options", "summer vibes", "summer weather"]):
            user_seasonal_preference = "summer"
        elif any(season in query_lower for season in ["spring", "spring options", "spring vibes"]):
            user_seasonal_preference = "spring"
        elif any(season in query_lower for season in ["fall", "autumn", "fall options"]):
            user_seasonal_preference = "fall"
        
        # Use user's seasonal preference if detected, otherwise current season
        season_for_justification = user_seasonal_preference or current_season
        
        if season_for_justification and any(seasonal_attrs.get('fabrics', [])):
            justification_parts.append(f"ideal for {season_for_justification} weather")
        
        # Price context
        price_parts = []
        if "price" in attributes:
            price_parts.append(f"under ${attributes['price']}")
        elif "price_max" in attributes:  # Also check for price_max alone
            price_parts.append(f"under ${attributes['price_max']}")
        elif "price_min" in attributes and "price_max" in attributes:
            if attributes['price_min'] == attributes['price_max']:
                price_parts.append(f"around ${attributes['price_min']}")
            else:
                price_parts.append(f"between ${attributes['price_min']} and ${attributes['price_max']}")
        elif "price_min" in attributes:  # Add handling for price_min alone
            price_parts.append(f"above ${attributes['price_min']}")
        
        # Combine parts into a natural sentence
        if justification_parts:
            if len(justification_parts) > 1:
                main_justification = f"These recommendations match {', '.join(justification_parts[:-1])} and {justification_parts[-1]}"
            else:
                main_justification = f"These recommendations match {justification_parts[0]}"
        else:
            main_justification = "These items are curated based on your preferences"
        
        if price_parts:
            main_justification += f", all {price_parts[0]}"
        
        # Add trending note if applicable
        trending_note = ""
        trending_keywords = ['oversized', 'cropped', 'high waisted']
        if any(keyword in original_query.lower() for keyword in trending_keywords):
            trending_note = " We've also included some trending pieces that match your vibe!"
        
        return main_justification + "." + trending_note

    def get_current_season(self):
        """Get current season name"""
        month = datetime.now().month
        if month in [3, 4, 5]:
            return 'spring'
        elif month in [6, 7, 8]:
            return 'summer'
        elif month in [9, 10, 11]:
            return 'fall'
        else:
            return 'winter'

    def generate_smart_follow_up_questions(self, missing_attributes, user_query):
        """Generate contextually relevant follow-up questions using advanced features"""
        questions = self.conversation_manager.suggest_smart_questions(
            missing_attributes, 
            self.conversation_manager.user_preferences.get('default_user', {})
        )
        
        if questions:
            return questions
        
        # Fallback to enhanced question generation
        enhanced_questions = []
        
        for attr in missing_attributes[:2]:  # Limit to 2 questions
            if attr == 'available_sizes':
                # Check if we've learned the user's size before
                learned_size = self.conversation_manager.get_learned_size('default_user', 
                                                                         self.conversation_attributes.get('category', 'general'))
                if learned_size:
                    enhanced_questions.append(f"Should I use your usual size {learned_size}, or are you looking for a different size?")
                else:
                    enhanced_questions.append("What size are you looking for?")
            elif attr == 'price':
                enhanced_questions.append("What's your budget range for this piece?")
            elif attr == 'occasion':
                # Use seasonal context
                season = self.get_current_season()
                if season == 'summer':
                    enhanced_questions.append("Is this for a specific summer occasion (beach, BBQ, vacation) or everyday wear?")
                elif season == 'winter':
                    enhanced_questions.append("What winter occasion are you shopping for (holiday party, work, cozy weekend)?")
                else:
                    enhanced_questions.append("What's the occasion you're shopping for?")
            elif attr == 'fit':
                enhanced_questions.append("How would you like it to fit - relaxed and comfy or more fitted and structured?")
            elif attr == 'color_or_print':
                enhanced_questions.append("Any color preferences or should I stick to versatile neutrals?")
        
        return enhanced_questions

    def apply_flexible_filtering(self, attributes, df_to_filter):
        """Apply flexible filtering that relaxes constraints to avoid zero results"""
        filtered_df = df_to_filter.copy() if df_to_filter is not None else self.apparel_data.copy()
        print(f"Starting flexible filtering with {len(filtered_df)} items")
        
        # Priority order for applying filters (most important first)
        filter_priority = ['category', 'pant_type', 'price', 'available_sizes', 'fit', 'fabric', 'occasion', 'color_or_print']
        
        for attr in filter_priority:
            if attr not in attributes:
                continue
                
            value = attributes[attr]
            if value is None:
                continue
            
            # Apply filter and check if we still have reasonable results
            temp_df = self.apply_single_filter(filtered_df, attr, value)
            
            # If applying this filter results in too few items, be more flexible
            if len(temp_df) == 0:
                print(f"Filter {attr}={value} eliminated all results, trying flexible approach...")
                temp_df = self.apply_flexible_single_filter(filtered_df, attr, value)
            
            if len(temp_df) > 0:
                filtered_df = temp_df
                print(f"After {attr} filter: {len(filtered_df)} items")
            else:
                print(f"Skipping {attr} filter to avoid zero results")
        
        return filtered_df
    
    def apply_single_filter(self, df, attr, value):
        """Apply a single filter exactly as in the main filter_apparel method"""
        filtered_df = df.copy()
        
        if attr == 'available_sizes' or attr == 'size':
            if 'available_sizes' in filtered_df.columns:
                if isinstance(value, list):
                    size_pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                    filtered_df = filtered_df[filtered_df['available_sizes'].astype(str).str.contains(size_pattern, case=False, na=False)]
                else:
                    filtered_df = filtered_df[filtered_df['available_sizes'].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
        
        elif attr == 'price' or attr == 'price_max':
            if 'price' in filtered_df.columns:
                filtered_df = filtered_df[pd.to_numeric(filtered_df['price'], errors='coerce') <= value]
        
        elif attr == 'pant_type':
            if isinstance(value, list):
                pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains(pattern, case=False, na=False)]
            else:
                if value.lower() in ["cargo", "cargos"]:
                    filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains("cargo", case=False, na=False)]
                elif value.lower() in ["denim", "jean", "jeans"]:
                    filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains("denim|jean", case=False, na=False)]
                else:
                    filtered_df = filtered_df[filtered_df['name'].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
        
        elif attr in filtered_df.columns:
            if isinstance(value, list):
                pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(pattern, case=False, na=False)]
            else:
                filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
        
        return filtered_df
    
    def apply_flexible_single_filter(self, df, attr, value):
        """Apply a more flexible version of a filter"""
        filtered_df = df.copy()
        
        # For occasion, try partial matches or related occasions
        if attr == 'occasion':
            if isinstance(value, list):
                all_alternatives = []
                for v in value:
                    occasion_alternatives = {
                        'Everyday': ['Casual', 'Work', 'Versatile'],
                        'Work': ['Everyday', 'Professional', 'Casual'],
                        'Party': ['Evening', 'Special Event', 'Night Out'],
                        'Casual': ['Everyday', 'Weekend', 'Relaxed']
                    }
                    alternatives = occasion_alternatives.get(v, [v])
                    all_alternatives.extend(alternatives)
                all_alternatives = list(set(all_alternatives))  # Remove duplicates
            else:
                occasion_alternatives = {
                    'Everyday': ['Casual', 'Work', 'Versatile'],
                    'Work': ['Everyday', 'Professional', 'Casual'],
                    'Party': ['Evening', 'Special Event', 'Night Out'],
                    'Casual': ['Everyday', 'Weekend', 'Relaxed']
                }
                all_alternatives = occasion_alternatives.get(value, [value])
            
            pattern = '|'.join([re.escape(str(alt)) for alt in all_alternatives])
            filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(pattern, case=False, na=False)]
        
        # For fabric, try broader categories
        elif attr == 'fabric':
            if isinstance(value, list):
                # If looking for natural fibers, include more options
                natural_fibers = ['Cotton', 'Linen', 'Silk', 'Wool']
                synthetic_fibers = ['Polyester', 'Rayon', 'Nylon']
                
                expanded_fabrics = []
                for v in value:
                    expanded_fabrics.append(v)
                    if v in natural_fibers:
                        expanded_fabrics.extend([f for f in natural_fibers if f != v])
                
                pattern = '|'.join([re.escape(str(f).strip()) for f in set(expanded_fabrics)])
                filtered_df = filtered_df[filtered_df['fabric'].astype(str).str.contains(pattern, case=False, na=False)]
            else:
                # Use fuzzy matching for single fabric
                fabric_matches = self.smart_filtering.fuzzy_match_attributes(str(value), filtered_df['fabric'].unique())
                if fabric_matches:
                    best_matches = [match[0] for match in fabric_matches[:3]]  # Top 3 matches
                    pattern = '|'.join([re.escape(match) for match in best_matches])
                    filtered_df = filtered_df[filtered_df['fabric'].astype(str).str.contains(pattern, case=False, na=False)]
        
        # For fit, try related fits
        elif attr == 'fit':
            if isinstance(value, list):
                all_alternatives = []
                for v in value:
                    fit_alternatives = {
                        'Relaxed': ['Loose', 'Comfortable', 'Flowy'],
                        'Fitted': ['Tailored', 'Slim', 'Body hugging'],
                        'Tailored': ['Fitted', 'Structured', 'Professional']
                    }
                    alternatives = fit_alternatives.get(v, [v])
                    all_alternatives.extend(alternatives)
                all_alternatives = list(set(all_alternatives))  # Remove duplicates
            else:
                fit_alternatives = {
                    'Relaxed': ['Loose', 'Comfortable', 'Flowy'],
                    'Fitted': ['Tailored', 'Slim', 'Body hugging'],
                    'Tailored': ['Fitted', 'Structured', 'Professional']
                }
                all_alternatives = fit_alternatives.get(value, [value])
            
            pattern = '|'.join([re.escape(str(alt)) for alt in all_alternatives])
            filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(pattern, case=False, na=False)]
        
        # For other attributes, just try case-insensitive partial matching
        elif attr in filtered_df.columns:
            if isinstance(value, list):
                pattern = '|'.join([re.escape(str(v)) for v in value])
                filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(pattern, case=False, na=False)]
            else:
                filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(str(value), case=False, na=False)]
        
        return filtered_df

    def reset_conversation_state(self):
        """Reset conversation state for a fresh start."""
        self.conversation_attributes = {}
        self.follow_up_turns_count = 0
        self.no_preference_attributes = []
        self.last_follow_up_attribute = None
        print("Conversation state reset for new query.")
        return True  # Return flag indicating reset occurred
    
    def should_reset_conversation(self, query: str) -> bool:
        """Determine if this is a new conversation that should reset state."""
        # Only reset when user explicitly mentions "reset"
        query_lower = query.lower().strip()
        
        # Only check for the explicit "reset" keyword as specified in the requirements
        return "reset" in query_lower
