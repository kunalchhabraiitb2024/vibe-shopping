import json
import re
import os
import pandas as pd
from datetime import datetime
from dotenv import load_dotenv
import google.generativeai as genai
from keyword_mappings import get_keyword_attributes, search_keywords
from dynamic_keyword_mapper import DynamicKeywordMapper, get_dynamic_keywords
from advanced_enhancements import (
    AdvancedVibeEngine, SmartFiltering, ConversationFlowManager, 
    AdvancedRecommendationEngine,
    apply_advanced_filtering, generate_enhanced_recommendations
)

load_dotenv()

class VibeShoppingAgent:
    def __init__(self, apparel_xlsx_path=os.path.join(os.path.dirname(__file__), 'Apparels_shared.xlsx'), 
                vibe_mapping_path=os.path.join(os.path.dirname(__file__), 'vibe_to_attribute.txt')):
        self.apparel_data = self.load_apparel_data(apparel_xlsx_path)
        self.vibe_mappings = self.load_vibe_mappings(vibe_mapping_path)
        self.important_attributes = ['category', 'available_sizes', 'price', 'fit', 'fabric', 'sleeve_length', 'occasion']
        self.conversation_attributes = {}
        self.follow_up_turns_count = 0
        self.no_preference_attributes = []
        self.last_follow_up_attribute = None
        self.previous_conversation_attributes = {}
        self.current_query = ""

        self.vibe_engine = AdvancedVibeEngine()
        self.smart_filtering = SmartFiltering()
        self.conversation_manager = ConversationFlowManager()
        
        self.dynamic_mapper = DynamicKeywordMapper(apparel_xlsx_path)
        print(f"Dynamic keyword mapper initialized with catalog keywords.")
        
        api_key = os.getenv('GOOGLE_API_KEY')
        if not api_key:
            print("Error: GOOGLE_API_KEY environment variable not set.")
            print("Please set the GOOGLE_API_KEY environment variable with your Gemini API key.")
            self.model = None
        else:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-2.0-flash-exp')
            print("Google Gemini 2.0 Flash model initialized with advanced enhancements.")

        self.recommendation_engine = AdvancedRecommendationEngine(self.apparel_data)

    def load_apparel_data(self, data_path):
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
        mappings = {}
        try:
            with open(text_path, 'r') as f:
                content = f.read()
                
                pattern = r'"([^"]+)":\s*(\{[^}]+\})'
                matches = re.findall(pattern, content, re.MULTILINE | re.DOTALL)
                
                for vibe_name, json_str in matches:
                    try:
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
        if not self.model:
            print("LLM model not initialized due to missing API key.")
            fallback_attrs = self.dynamic_mapper._fallback_keyword_matching(query) if hasattr(self.dynamic_mapper, '_fallback_keyword_matching') else self.fallback_attribute_detection(query)

            no_more_questions = any(phrase in query.lower() for phrase in [
                "no preference", "doesn't matter", "don't care", "anything works", "anything", "any"
            ])
            return fallback_attrs, no_more_questions

        try:
            dynamic_result = get_dynamic_keywords(query, last_asked_attribute=last_asked_attribute)
            if dynamic_result:
                print(f"Dynamic mapper result (with last_asked_attribute='{last_asked_attribute}'): {dynamic_result}")
                
                extracted_attributes = dynamic_result
                
                if (last_asked_attribute == 'available_sizes' and 
                    'fit' in extracted_attributes and 
                    'available_sizes' not in extracted_attributes):
                    
                    print(f"Size was asked but got fit attribute. Trying fallback size detection...")
                    fallback_attrs = self.fallback_attribute_detection(query)
                    if 'available_sizes' in fallback_attrs:
                        extracted_attributes = fallback_attrs
                        print(f"Fallback detected size: {extracted_attributes}")
                
                no_more_questions = any(phrase in query.lower() for phrase in [
                    "no preference", "doesn't matter", "don't care", "anything works", 
                    "anything", "anything is fine", "all good", "i'm flexible"
                ])
                
                return extracted_attributes, no_more_questions
        except Exception as e:
            print(f"Dynamic keyword mapping failed: {e}")
        
        fallback_attrs = self.fallback_attribute_detection(query)
        no_more_questions = any(phrase in query.lower() for phrase in [
            "no preference", "doesn't matter", "don't care", "anything works", "anything", "any"
        ])
        return fallback_attrs, no_more_questions
        
        seasonal_context = self.vibe_engine.get_seasonal_attributes()
        style_combinations = self.vibe_engine.analyze_style_combination(query)
        catalog_keywords = self.dynamic_mapper.get_catalog_keywords()

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

            return all_identified_attributes, no_more_questions

        except Exception as e:
            print(f"Error calling LLM or parsing response: {e}")
            fallback_attrs = self.dynamic_mapper.fallback_keyword_detection(query)
            no_more_questions = any(phrase in query.lower() for phrase in [
                "no preference", "doesn't matter", "don't care", "anything works", "anything", "any"
            ])
            return fallback_attrs, no_more_questions

    def process_query(self, query):
        print(f"Processing query: {query}")
        
        self.current_query = query
        self.previous_conversation_attributes = self.conversation_attributes.copy()
        
        if self.should_reset_conversation(query):
            self.reset_conversation_state()
            return {
                "response_type": "message", 
                "message": "I've reset our conversation. What are you looking for today?",
                "attributes": {}
            }
        
        no_more_questions = any(phrase in query.lower() for phrase in [
            "no follow up", "no more questions", "just show me", "show results", 
            "no preference", "doesn't matter", "don't care", "anything works", 
            "anything", "skip questions", "no questions"
        ])
        
        identified_attributes, llm_no_more_questions = self.get_attributes_from_llm(
            query, self.conversation_attributes, self.last_follow_up_attribute
        )
        
        no_more_questions = no_more_questions or llm_no_more_questions
        print(f"Extracted attributes: {identified_attributes}")
        print(f"No more questions requested: {no_more_questions}")
        
        if identified_attributes:
            for attr, value in identified_attributes.items():
                if value is not None:
                    self.conversation_attributes[attr] = value
                    print(f"Updated {attr}: {value}")
        
        if 'pant_type' in self.conversation_attributes and 'category' not in self.conversation_attributes:
            self.conversation_attributes['category'] = 'pants'
            print("Inferred category as 'pants' from pant_type")
        
        print(f"Current conversation state: {self.conversation_attributes}")
        
        should_check_compatibility = (
            not self.conversation_attributes and
            not self.is_follow_up_or_correction(query) and
            self.looks_like_product_search(query)
        )
        
        if should_check_compatibility:
            is_compatible, suggested_alternatives, explanation = self.validate_catalog_compatibility(query)
            if not is_compatible:
                alternative_message = ""
                if suggested_alternatives:
                    alternative_message = f" However, we have a great selection of {', '.join(suggested_alternatives)} that might interest you!"
                
                response_message = f"I'm sorry, but {explanation}{alternative_message} Would you like to explore what we have available?"
                self.reset_conversation_state()
                
                return {
                    "response_type": "message",
                    "message": response_message,
                    "attributes": {},
                    "suggested_alternatives": suggested_alternatives
                }
        
        if self.conversation_attributes:
            filtered_df = self.filter_apparel(self.conversation_attributes)
            print(f"Filtered to {len(filtered_df)} items based on current attributes")
        else:
            filtered_df = self.apparel_data.copy()
            print(f"No filters applied yet, working with full catalog ({len(filtered_df)} items)")
        
        should_ask_followup = False
        follow_up_question = None
        
        if not no_more_questions and self.follow_up_turns_count < 2:
            if len(filtered_df) > 15:
                should_ask_followup = True
            elif len(filtered_df) == 0 and len(self.conversation_attributes) > 0:
                should_ask_followup = True
            elif not self.conversation_attributes:
                should_ask_followup = True
        
        if should_ask_followup:
            missing_important_attrs = []
            for attr in ['category', 'available_sizes', 'price', 'fit', 'occasion']:
                if attr not in self.conversation_attributes:
                    missing_important_attrs.append(attr)
            
            if missing_important_attrs:
                attr_to_ask = missing_important_attrs[0]
                
                if attr_to_ask == 'category':
                    follow_up_question = "What type of clothing are you looking for? (e.g., 'dresses', 'tops', 'pants' or 'no preference')"
                elif attr_to_ask == 'available_sizes':
                    follow_up_question = "What size do you need? (e.g., 'M', 'L to XL', 'at least Large' or 'no preference')"
                elif attr_to_ask == 'price':
                    follow_up_question = "What's your budget? (e.g., 'under $100', '$50-150', 'at least $30' or 'no budget range')"
                elif attr_to_ask == 'fit':
                    follow_up_question = "How would you like it to fit? (e.g., 'relaxed', 'tailored to loose', 'at least fitted' or 'no preference')"
                elif attr_to_ask == 'occasion':
                    follow_up_question = "What's the occasion? (e.g., 'work', 'casual to formal', 'at least semi-formal' or 'no preference')"
                
                self.last_follow_up_attribute = attr_to_ask
        
        if follow_up_question and not no_more_questions:
            self.follow_up_turns_count += 1
            print(f"Asking follow-up question about {self.last_follow_up_attribute}")
            return {
                "response_type": "follow_up",
                "questions": [follow_up_question],
                "attributes": self.conversation_attributes
            }
        else:
            print("Proceeding to show results...")
            self.last_follow_up_attribute = None
            
            if len(filtered_df) == 0 and self.conversation_attributes:
                print("No results found, attempting constraint relaxation...")
                filtered_df = self.apply_flexible_filtering(self.conversation_attributes, self.apparel_data)
                print(f"After relaxation: {len(filtered_df)} items found")
            
            filtered_df = apply_advanced_filtering(filtered_df, query, self.smart_filtering)
            enhanced_recommendations = generate_enhanced_recommendations(self, filtered_df, self.conversation_attributes)
            
            recommended_items = []
            for item in enhanced_recommendations[:7]:
                rec_item = {
                    'id': item.get('id', '') if isinstance(item, dict) else str(getattr(item, 'id', '')),
                    'name': item.get('name', '') if isinstance(item, dict) else getattr(item, 'name', ''),
                    'price': item.get('price', 0) if isinstance(item, dict) else getattr(item, 'price', 0),
                    'is_trending': item.get('is_trending', False) if isinstance(item, dict) else getattr(item, 'is_trending', False)
                }
                recommended_items.append(rec_item)
            
            justification = self.generate_enhanced_justification(
                self.conversation_attributes, 
                query, 
                num_results=len(recommended_items)
            )
            
            return {
                "response_type": "recommendation",
                "recommendations": recommended_items,
                "justification": justification,
                "attributes": self.conversation_attributes,
                "enhanced_features": True
            }

    def extract_attributes_from_query_keyword(self, query):
        extracted = {}
        query_lower = query.lower()

        if "dress" in query_lower or "dresses" in query_lower:
            extracted["category"] = ["dress"]
        if "top" in query_lower or "tops" in query_lower:
            if "category" in extracted:
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
                  extracted["category"].append("pants")
             else:
                  extracted["category"] = ["pants"]
        if "pant" in query_lower or "pants" in query_lower:
             if "category" in extracted:
                  extracted["category"].append("pants")
             else:
                  extracted["category"] = ["pants"]

        size_match = re.search(r'size\s+([xsmlXSML]+)', query_lower)
        if size_match:
            extracted["size"] = [size_match.group(1).upper()]

        price_match_range = re.search(r'between\s*\$?(\d+)[kK]?\s*-\s*\$?(\d+)[kK]?', query_lower)
        price_match_under = re.search(r'under\s*\$?(\d+)[kK]?', query_lower)
        price_match_above = re.search(r'above\s*\$?(\d+)[kK]?', query_lower)
        price_match_exactly = re.search(r'\$(\d+)[kK]?', query_lower)
        price_match_budget = re.search(r'budget\s+is.*?(\d+)\$?[kK]?', query_lower)
        price_match_trailing = re.search(r'(\d+)\$[kK]?', query_lower)

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

        if "sleeveless" in query_lower:
            extracted["sleeve_length"] = "Sleeveless"
        if "short sleeve" in query_lower or "short-sleeve" in query_lower:
             extracted["sleeve_length"] = "Short sleeves"
        if "long sleeve" in query_lower or "long-sleeve" in query_lower:
             extracted["sleeve_length"] = "Long sleeves"

        if "relaxed fit" in query_lower or "flowy" in query_lower:
            extracted["fit"] = "Relaxed"
        if "bodycon" in query_lower or "body hugging" in query_lower:
            extracted["fit"] = "Body hugging"
        if "tailored fit" in query_lower:
            extracted["fit"] = "Tailored"

        if "linen" in query_lower: extracted["fabric"] = ["Linen"]
        if "cotton" in query_lower: 
             if "fabric" in extracted:
                  extracted["fabric"].append("Cotton")
             else:
                  extracted["fabric"] = ["Cotton"]
        fabric_matches = re.findall(r'(linen|cotton|silk|velvet|satin)', query_lower)
        if fabric_matches:
             if "fabric" not in extracted:
                  extracted["fabric"] = []
             for fabric in fabric_matches:
                 capitalized_fabric = fabric.capitalize()
                 if capitalized_fabric not in extracted["fabric"]:
                      extracted["fabric"].append(capitalized_fabric)

        if "brunch" in query_lower: extracted["occasion"] = "Everyday"
        if "party" in query_lower: extracted["occasion"] = "Party"
        if "vacation" in query_lower or "vacay" in query_lower: extracted["occasion"] = "Vacation"
        if "work" in query_lower or "office" in query_lower: extracted["occasion"] = "Work"

        return extracted

    def identify_vibes_from_query_keyword(self, query):
        identified = []
        query_lower = query.lower()
        
        for vibe_key in self.vibe_mappings.keys():
             if vibe_key in query_lower:
                  identified.append(vibe_key)
        
        individual_vibe_words = ["cute", "elevated", "comfy", "polish", "flowy", "glamorous", "beachy", "retro"]
        for word in individual_vibe_words:
             if word in query_lower and word not in ''.join(identified):
                  identified.append(word)

        identified = list(dict.fromkeys(identified))
        print(f"Identified Vibes (Keyword): {identified}")
        return identified
        

    def infer_attributes_from_vibes(self, vibes):
        inferred = {}
        for vibe in vibes:
            if vibe in self.vibe_mappings:
                for attr, value in self.vibe_mappings[vibe].items():
                     if isinstance(value, list) and attr in inferred and isinstance(inferred[attr], list):
                          inferred[attr].extend(value)
                          inferred[attr] = list(dict.fromkeys(inferred[attr]))
                     else:
                          inferred[attr] = value
        return inferred

    def determine_follow_up_questions(self, current_attributes):
        questions = []
        asked_count = 0
        
        print(f"Determining follow-up questions based on attributes: {current_attributes}")

        attributes_to_ask = ['category', 'size', 'price_max', 'fit', 'fabric', 'sleeve_length', 'occasion']
        
        if asked_count < 2 and 'size' not in current_attributes and ('price_max' not in current_attributes and 'price_min' not in current_attributes):
             questions.append("Any must-haves like size or budget range to keep in mind? (e.g., 'M size, under $100', 'L to XL, $50-150 range', 'at least Large, no budget' or 'no preference')")
             asked_count += 1
        
        for attr in attributes_to_ask:
            if attr in ['size', 'price_max'] and len(questions) > 0 and "Any must-haves like size or budget range to keep in mind?" in questions[0]:
                 continue
            
            if attr == 'price_max' and 'price_min' in current_attributes and 'price_max' not in current_attributes:
                 if asked_count < 2:
                     questions.append(f"Okay, you mentioned a price around ${current_attributes['price_min']}. Do you have a maximum budget in mind? (e.g., 'under $200', 'up to $150', 'at least $100' or 'no max budget')")
                     asked_count += 1
                 continue
            elif attr == 'price_max' and 'price_max' in current_attributes and 'price_min' not in current_attributes:
                 if asked_count < 2:
                      questions.append(f"Okay, you mentioned a maximum price of ${current_attributes['price_max']}. Do you have a minimum budget in mind? (e.g., 'at least $30', '$20-50 range', 'under $40' or 'no minimum')")
                      asked_count += 1
                 continue

            if asked_count < 2 and attr not in current_attributes and attr != 'price_min':
                if attr == 'category':
                    questions.append("Do you have a preference between dresses, tops & skirts, or something more casual like jeans? (e.g., 'dresses', 'tops', 'casual to formal' or 'no preference')")
                elif attr == 'size':
                     size_value = current_attributes.get('size')
                     if not size_value or (isinstance(size_value, str) and size_value.lower() == 'any') or (isinstance(size_value, list) and len(size_value) == 0) or (isinstance(size_value, list) and size_value[0].lower() == 'any'):
                         questions.append("What size are you looking for? (e.g., 'M', 'L to XL', 'at least Large' or 'no preference')")
                elif attr == 'fit':
                     questions.append("What kind of fit do you prefer? (e.g., 'relaxed', 'tailored to loose', 'at least fitted' or 'no preference')")
                elif attr == 'fabric':
                     questions.append("Any preference on fabric? (e.g., 'cotton', 'silk to linen', 'at least breathable' or 'no preference')")
                elif attr == 'sleeve_length':
                     questions.append("What sleeve length are you looking for? (e.g., 'short', 'long to sleeveless', 'at least 3/4 sleeve' or 'no preference')")
                elif attr == 'occasion':
                     questions.append("What's the occasion? (e.g., 'work', 'casual to formal', 'at least semi-formal' or 'no preference')")
                asked_count += 1

        return questions

    def filter_apparel(self, attributes, df_to_filter=None):
        filtered_df = df_to_filter.copy() if df_to_filter is not None else self.apparel_data.copy()
        print(f"Initial items before filtering in filter_apparel: {len(filtered_df)}")
        print(f"DEBUG: All attributes being used for filtering: {attributes}")
        
        if not attributes:
            seasonal_attrs = self.vibe_engine.get_seasonal_attributes()
            print(f"No attributes provided, applying seasonal intelligence: {seasonal_attrs}")
        
        for attr, value in attributes.items():
            if value is None:
                print(f"Skipping filter for '{attr}' as value is None.")
                continue

            print(f"Applying filter: {attr} = {value}")
            
            if attr == 'available_sizes' or attr == 'size':
                if 'available_sizes' in filtered_df.columns:
                    if isinstance(value, list):
                        size_pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                        filtered_df = filtered_df[filtered_df['available_sizes'].astype(str).str.contains(size_pattern, case=False, na=False)]
                    else:
                        filtered_df = filtered_df[filtered_df['available_sizes'].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
                    print(f"Items after size filter: {len(filtered_df)}")
                continue
                
            if attr == 'price' or attr == 'price_max':
                if 'price' in filtered_df.columns:
                    print(f"DEBUG: Filtering price with value {value}")
                    print(f"DEBUG: Price column sample before filtering: {filtered_df['price'].head().tolist()}")
                    
                    filtered_df = filtered_df[pd.to_numeric(filtered_df['price'], errors='coerce') <= value]
                    print(f"Items after price filter: {len(filtered_df)}")
                    
                    if len(filtered_df) > 0:
                        print(f"DEBUG: Price column sample after filtering: {filtered_df['price'].head().tolist()}")
                    else:
                        print(f"DEBUG: No items match the price filter of {value}")
                continue
            
            if attr == 'color_or_print' or attr == 'color':
                if 'color_or_print' in filtered_df.columns:
                    filtered_df = self.smart_filtering.match_color_palette(str(value), filtered_df)
                    print(f"Items after enhanced color filter: {len(filtered_df)}")
                continue
            
            elif attr == 'fabric':
                if 'fabric' in filtered_df.columns:
                    if isinstance(value, list):
                        pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                        filtered_df = filtered_df[filtered_df['fabric'].astype(str).str.contains(pattern, case=False, na=False)]
                    else:
                        fabric_matches = self.smart_filtering.fuzzy_match_attributes(str(value), filtered_df['fabric'].unique())
                        if fabric_matches and fabric_matches[0][1] > 0.7:
                            best_match = fabric_matches[0][0]
                            filtered_df = filtered_df[filtered_df['fabric'].astype(str).str.contains(re.escape(best_match), case=False, na=False)]
                            print(f"Used fuzzy matching: '{value}' -> '{best_match}'")
                        else:
                            filtered_df = filtered_df[filtered_df['fabric'].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
                    print(f"Items after enhanced fabric filter: {len(filtered_df)}")
                continue
            
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
                print(f"Items after {attr} filter (searching name column): {len(filtered_df)}")
                print(f"DEBUG: Searching for '{value}' in name column, found {len(filtered_df)} items")
            
            elif attr in filtered_df.columns:
                if isinstance(value, list):
                    pattern = '|'.join([re.escape(str(v).strip()) for v in value])
                    if attr == 'category':
                        filtered_df = filtered_df[filtered_df[attr].astype(str).str.strip().str.contains(pattern, case=False, na=False)]
                    else:
                        filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(pattern, case=False, na=False)]
                    print(f"Items after {attr} filter: {len(filtered_df)}")
                else:
                    if attr == 'category':
                        filtered_df = filtered_df[filtered_df[attr].astype(str).str.strip().str.contains(re.escape(str(value).strip()), case=False, na=False)]
                    else:
                        filtered_df = filtered_df[filtered_df[attr].astype(str).str.contains(re.escape(str(value).strip()), case=False, na=False)]
                    print(f"Items after {attr} filter: {len(filtered_df)}")
        
        print(f"Final items after filtering in filter_apparel: {len(filtered_df)}")
        return filtered_df

    def generate_justification(self, attributes):
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
            
        return "Based on " + ", ".join(justification_parts) + ", I found these perfect matches for you."

    def fallback_attribute_detection(self, query):
        """Enhanced fallback method using comprehensive keyword mapping system"""
        detected = {}
        query_lower = query.lower().strip()
        
        # PRIORITY: Check for size words first (especially important for follow-up responses)
        word_size_mapping = {
            'small': 'S',
            'medium': 'M', 
            'large': 'L',
            'extra large': 'XL',
            'extra-large': 'XL',
            'xlarge': 'XL',
            'x-large': 'XL'
        }
        
        # Check if the entire query is just a size word (common in follow-up responses)
        if query_lower.strip() in word_size_mapping:
            detected["available_sizes"] = word_size_mapping[query_lower.strip()]
            print(f"[DEBUG] Direct size word mapping: '{query_lower.strip()}' → {word_size_mapping[query_lower.strip()]}")
            return detected
        
        # Check for size words within the query
        for word_size, letter_size in word_size_mapping.items():
            if word_size in query_lower:
                detected["available_sizes"] = letter_size
                print(f"[DEBUG] Size word found: '{word_size}' → {letter_size}")
                return detected  # Return immediately to avoid other interpretations
        
        # Letter-based size patterns
        size_patterns = [r'\b([smlxSMLX]+)\b', r'size\s+([smlxSMLX]+)', r'([smlxSMLX]+)\s+size']
        for pattern in size_patterns:
            size_match = re.search(pattern, query_lower)
            if size_match:
                detected["available_sizes"] = size_match.group(1).upper()
                print(f"[DEBUG] Letter size pattern: {size_match.group(1).upper()}")
                return detected
        
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
        
        # Use the comprehensive keyword mapping system (only if no size detected)
        if not detected:
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
        
        # Price detection (only if no size detected)
        if not detected or "available_sizes" not in detected:
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

    def generate_enhanced_justification(self, attributes, original_query, num_results=0):
        justification_parts = []
        filter_changes = []
        
        if self.previous_conversation_attributes:
            for attr, value in attributes.items():
                prev_value = self.previous_conversation_attributes.get(attr)
                if prev_value != value:
                    if prev_value is None:
                        filter_changes.append(f"added {attr}: {value}")
                    else:
                        filter_changes.append(f"updated {attr}: {prev_value} → {value}")
        else:
            for attr, value in attributes.items():
                filter_changes.append(f"set {attr}: {value}")
        
        seasonal_attrs = self.vibe_engine.get_seasonal_attributes()
        current_season = self.get_current_season()
        style_combinations = self.vibe_engine.analyze_style_combination(original_query)
        
        active_filters = []
        
        if "category" in attributes:
            active_filters.append(f"{attributes['category']} items")
            
        if "pant_type" in attributes:
            active_filters.append(f"specifically {attributes['pant_type']} style")
        
        if style_combinations:
            style_names = list(style_combinations.keys())
            if len(style_names) == 1:
                active_filters.append(f"{style_names[0]} style")
            else:
                active_filters.append(f"{' and '.join(style_names)} aesthetic")
        
        if "fit" in attributes:
            if isinstance(attributes['fit'], list):
                active_filters.append(f"{' or '.join(attributes['fit'])} fit")
            else:
                active_filters.append(f"{attributes['fit']} fit")
        
        if "available_sizes" in attributes:
            size = attributes['available_sizes']
            if isinstance(size, list):
                active_filters.append(f"available in sizes {', '.join(size)}")
            else:
                active_filters.append(f"available in size {size}")
        
        price_constraint = ""
        if "price" in attributes:
            price_constraint = f"under ${attributes['price']}"
        elif "price_max" in attributes:
            price_constraint = f"under ${attributes['price_max']}"
        elif "price_min" in attributes and "price_max" in attributes:
            if attributes['price_min'] == attributes['price_max']:
                price_constraint = f"around ${attributes['price_min']}"
            else:
                price_constraint = f"between ${attributes['price_min']}-${attributes['price_max']}"
        elif "price_min" in attributes:
            price_constraint = f"above ${attributes['price_min']}"
        
        if price_constraint:
            active_filters.append(price_constraint)
        
        if "occasion" in attributes:
            if isinstance(attributes['occasion'], list):
                active_filters.append(f"perfect for {' and '.join(attributes['occasion'])} occasions")
            else:
                active_filters.append(f"perfect for {attributes['occasion']} occasions")
        
        if "fabric" in attributes:
            if isinstance(attributes['fabric'], list):
                active_filters.append(f"made with {' and '.join(attributes['fabric'])} fabrics")
            else:
                active_filters.append(f"crafted in {attributes['fabric']}")
                
        if "color_or_print" in attributes:
            active_filters.append(f"in {attributes['color_or_print']} tones")
        
        # Seasonal context
        user_seasonal_preference = None
        query_lower = original_query.lower()
        if any(season in query_lower for season in ["winter", "winter options", "winter vibes"]):
            user_seasonal_preference = "winter"
        elif any(season in query_lower for season in ["summer", "summer options", "summer vibes"]):
            user_seasonal_preference = "summer"
        elif any(season in query_lower for season in ["spring", "spring options", "spring vibes"]):
            user_seasonal_preference = "spring"
        elif any(season in query_lower for season in ["fall", "autumn", "fall options"]):
            user_seasonal_preference = "fall"
        
        season_for_justification = user_seasonal_preference or current_season
        if season_for_justification and any(seasonal_attrs.get('fabrics', [])):
            active_filters.append(f"ideal for {season_for_justification} weather")
        
        # Build the main justification
        if active_filters:
            if len(active_filters) == 1:
                filter_description = active_filters[0]
            elif len(active_filters) == 2:
                filter_description = f"{active_filters[0]} and {active_filters[1]}"
            else:
                filter_description = f"{', '.join(active_filters[:-1])}, and {active_filters[-1]}"
        else:
            filter_description = "your preferences"
        
        # Customize message based on number of results
        if num_results == 0:
            main_justification = f"No items found matching {filter_description}"
        elif num_results == 1:
            main_justification = f"Found 1 perfect match for {filter_description}"
        else:
            main_justification = f"These {num_results} recommendations match {filter_description}"
        
        # Add filter change information for better observability
        change_description = ""
        if filter_changes:
            if len(filter_changes) == 1:
                change_description = f" (Filter update: {filter_changes[0]})"
            else:
                change_description = f" (Filter updates: {', '.join(filter_changes)})"
        
        # Add trending note if applicable
        trending_note = ""
        trending_keywords = ['oversized', 'cropped', 'high waisted']
        if any(keyword in original_query.lower() for keyword in trending_keywords):
            trending_note = " Including some trending pieces that match your vibe!"
        
        final_justification = main_justification + change_description + "." + trending_note
        
        print(f"Generated enhanced justification: {final_justification}")
        if filter_changes:
            print(f"Filter changes detected: {filter_changes}")
        
        return final_justification

    def validate_catalog_compatibility(self, query):
        """
        Use Gemini AI to determine if the user's request is compatible with our catalog.
        Returns: (is_compatible: bool, suggested_alternatives: list, explanation: str)
        """
        if not self.model:
            # Fallback without AI - just check for obvious mismatches
            query_lower = query.lower()
            unavailable_items = ['coat', 'jacket', 'blazer', 'sweater', 'cardigan', 'hoodie', 'lingerie', 'underwear', 'bra', 'swimwear', 'shoes', 'accessories', 'jewelry', 'bag', 'hat']
            for item in unavailable_items:
                if item in query_lower:
                    return False, [], f"We don't currently carry {item}s in our catalog."
            return True, [], ""

        # Get available categories from catalog
        available_categories = self.apparel_data['category'].unique().tolist()
        
        prompt = f"""You are a smart fashion catalog assistant. Analyze if a user's clothing request is compatible with our available inventory.

AVAILABLE CATALOG CATEGORIES: {available_categories}
We specialize in: tops, dresses, skirts, and pants (no outerwear, undergarments, accessories, or shoes)

USER REQUEST: "{query}"

Your task:
1. Determine if the user's request can be reasonably fulfilled with our available categories
2. If not compatible, suggest the closest alternatives from our catalog
3. Provide a helpful explanation

COMPATIBILITY RULES:
✅ COMPATIBLE requests:
- "dress", "top", "skirt", "pants" → Direct matches
- "blouse", "shirt", "tee" → Can be fulfilled with "tops"
- "jeans", "trousers", "cargo pants" → Can be fulfilled with "pants" 
- "mini dress", "maxi dress", "cocktail dress" → Can be fulfilled with "dress"

❌ NOT COMPATIBLE requests:
- "coat", "jacket", "blazer" → Outerwear (we don't carry)
- "lingerie", "bra", "underwear" → Undergarments (we don't carry)
- "shoes", "boots", "sneakers" → Footwear (we don't carry)
- "jewelry", "accessories", "bags" → Accessories (we don't carry)
- "swimwear", "bikini" → Swimwear (we don't carry)

RESPONSE FORMAT (JSON only):
{{
    "is_compatible": true/false,
    "suggested_alternatives": ["category1", "category2"],
    "explanation": "Brief explanation of why compatible/incompatible and suggestions",
    "confidence": 0.0-1.0
}}

Respond with ONLY valid JSON:"""

        try:
            response = self.model.generate_content(prompt)
            response_text = response.text.strip()
            # Clean the response text to ensure it's valid JSON
            response_text = response_text.replace('```json', '').replace('```', '').strip()
            
            result = json.loads(response_text)
            is_compatible = result.get('is_compatible', True)
            suggested_alternatives = result.get('suggested_alternatives', [])
            explanation = result.get('explanation', '')
            
            print(f"Catalog compatibility check: compatible={is_compatible}, alternatives={suggested_alternatives}")
            return is_compatible, suggested_alternatives, explanation
            
        except Exception as e:
            print(f"Error in catalog compatibility check: {e}")
            # Fallback to simple keyword matching
            query_lower = query.lower()
            unavailable_items = ['coat', 'jacket', 'blazer', 'sweater', 'cardigan', 'hoodie', 'lingerie', 'underwear', 'bra', 'swimwear', 'shoes', 'accessories', 'jewelry', 'bag', 'hat']
            for item in unavailable_items:
                if item in query_lower:
                    return False, [], f"We don't currently carry {item}s in our catalog."
            return True, [], ""

    def reset_conversation_state(self):
        """Reset conversation state for a fresh start."""
        self.conversation_attributes = {}
        self.previous_conversation_attributes = {}
        self.follow_up_turns_count = 0
        self.no_preference_attributes = []
        self.last_follow_up_attribute = None
        self.current_query = ""
        print("Conversation state reset for new query.")
        return True  # Return flag indicating reset occurred
    
    def should_reset_conversation(self, query: str) -> bool:
        """Determine if this is a new conversation that should reset state."""
        query_lower = query.lower().strip()
        
        # Reset on explicit "reset" keyword
        if "reset" in query_lower:
            return True
            
        # Also reset if this looks like a completely new product search 
        # (not a follow-up or correction) and we have existing conversation state
        if (self.conversation_attributes and 
            self.looks_like_product_search(query) and 
            not self.is_follow_up_or_correction(query) and
            not any(phrase in query_lower for phrase in ["more", "other", "different", "also", "another"])):
            return True
            
        return False

    def get_current_season(self):
        """Determine current season based on current date"""
        from datetime import datetime
        month = datetime.now().month
        if month in [12, 1, 2]:
            return "winter"
        elif month in [3, 4, 5]:
            return "spring"
        elif month in [6, 7, 8]:
            return "summer"
        else:  # 9, 10, 11
            return "fall"
    
    def apply_flexible_filtering(self, attributes, initial_df):
        """Apply flexible filtering by relaxing constraints when no results found"""
        # Try removing the most restrictive attributes one by one
        relaxed_attributes = attributes.copy()
        
        # Order of attributes to relax (most specific to least specific)
        relaxation_order = ['fit', 'fabric', 'sleeve_length', 'color_or_print', 'occasion']
        
        for attr_to_relax in relaxation_order:
            if attr_to_relax in relaxed_attributes:
                # Try without this attribute
                temp_attributes = {k: v for k, v in relaxed_attributes.items() if k != attr_to_relax}
                temp_df = self.filter_apparel(temp_attributes, initial_df)
                
                if len(temp_df) > 0:
                    print(f"Relaxed '{attr_to_relax}' constraint - found {len(temp_df)} items")
                    return temp_df
        
        # If still no results, return items from the primary category only
        if 'category' in attributes:
            category_only = {'category': attributes['category']}
            result_df = self.filter_apparel(category_only, initial_df)
            if len(result_df) > 0:
                print(f"Fallback to category-only filtering - found {len(result_df)} items")
                return result_df
        
        # Last resort - return a sample of items from the catalog
        print("No matches found even with relaxed filtering - returning sample items")
        return initial_df.head(10)

    def generate_smart_follow_up_questions(self, missing_attributes, query=""):
        """Generate contextually relevant follow-up questions"""
        questions = []
        
        # Use enhanced conversation manager if available
        if hasattr(self, 'conversation_manager'):
            return self.conversation_manager.suggest_smart_questions(missing_attributes)
        
        # Fallback to simple question generation with guided response formats
        question_templates = {
            'category': "What type of clothing are you looking for? (e.g., 'dresses', 'tops', 'pants' or 'no preference')",
            'available_sizes': "What size do you need? (e.g., 'M', 'L to XL', 'at least Large' or 'no preference')",
            'price': "What's your budget? (e.g., 'under $100', '$50-150', 'at least $30' or 'no budget range')",
            'fit': "How would you like it to fit? (e.g., 'relaxed', 'tailored to loose', 'at least fitted' or 'no preference')",
            'fabric': "Any fabric preferences? (e.g., 'cotton', 'silk to linen', 'at least breathable' or 'no preference')",
            'sleeve_length': "What sleeve length? (e.g., 'short', 'long to sleeveless', 'at least 3/4 sleeve' or 'no preference')",
            'occasion': "What's the occasion? (e.g., 'work', 'casual to formal', 'at least semi-formal' or 'no preference')",
            'color_or_print': "Any color preferences? (e.g., 'blue', 'red to pink', 'at least neutral' or 'no preference')"
        }
        
        # Prioritize questions based on importance
        priority_map = {
            'category': 3,
            'occasion': 2,
            'fit': 2,
            'available_sizes': 1,
            'price': 1
        }
        
        sorted_missing = sorted(missing_attributes, 
                              key=lambda x: priority_map.get(x, 0), reverse=True)
        
        for attr in sorted_missing[:2]:  # Limit to 2 questions
            if attr in question_templates:
                questions.append(question_templates[attr])
        
        return questions

    def is_follow_up_or_correction(self, query):
        """Detect if a query is a follow-up response or correction rather than a new product search."""
        import re
        
        query_lower = query.lower().strip()
        
        # Strong follow-up indicators that suggest this is a response to a previous question
        strong_follow_up_indicators = [
            "you didn't ask", "u didn't ask", "you forgot", "what about", 
            "my size", "my budget", "my price", 
            "actually", "but", "however", "though", "also",
            "that's what i mean", "yes that's", "right", "correct",
            "perfect", "good", "great", "nice"
        ]
        
        # Simple confirmations (separate check to avoid conflicts)
        simple_confirmations = ["ok", "okay"]
        
        # Check for strong follow-up indicators first
        for indicator in strong_follow_up_indicators:
            if indicator in query_lower:
                return True
                
        # Check for simple confirmations only if the query is short and doesn't contain product search phrases
        product_search_phrases = ["looking for", "show me", "find me", "search for"]
        has_product_search = any(phrase in query_lower for phrase in product_search_phrases)
        
        if not has_product_search and len(query_lower.split()) <= 2:
            if any(conf in query_lower for conf in simple_confirmations):
                return True
        
        # Simple confirmations/negations (but only if they're short)
        if len(query_lower.split()) <= 2:
            simple_responses = ["yes", "no", "sure", "nope", "yeah", "yep"]
            if query_lower.strip() in simple_responses:
                return True
        
        # Check for other product search phrases that should NOT be follow-ups (already checked above)
        if has_product_search:
            return False
        
        # Check if it's just a size (single letter or word)
        if query_lower.strip() in ['xs', 's', 'm', 'l', 'xl', 'xxl', 'small', 'medium', 'large', 'extra large']:
            return True
            
        # Check if it's just a number (budget/price)
        if re.match(r'^\d+(\$|dollars?)?$', query_lower.strip()):
            return True
        
        # Check if it's a size phrase like "size medium", "size small", etc.
        if re.match(r'^size\s+(xs|s|m|l|xl|xxl|small|medium|large|extra\s+large)$', query_lower):
            return True
            
        # Check if it's a single color name (common follow-up response)
        single_colors = ["red", "blue", "green", "black", "white", "pink", "purple", "yellow", "orange", "brown", "gray", "grey", "navy", "beige", "gold", "silver"]
        if query_lower.strip() in single_colors:
            return True
            
        # Check if it's a single occasion or style word (common follow-up response) - but only if it's truly single word
        single_attributes = ["casual", "formal", "work", "party", "date", "professional", "comfortable", "tight", "loose", "fitted", "relaxed"]
        if len(query_lower.split()) == 1 and query_lower.strip() in single_attributes:
            return True
            
        # Check for specific guided response formats we recommend
        guided_response_patterns = [
            # Upper/lower bounds patterns
            r'\bunder\s+\$?\d+', r'\bbelow\s+\$?\d+', r'\bless\s+than\s+\$?\d+',
            r'\bat\s+least\s+\w+', r'\bminimum\s+\w+', r'\babove\s+\$?\d+',
            r'\bover\s+\$?\d+', r'\bmore\s+than\s+\$?\d+',
            
            # Range patterns  
            r'\$?\d+\s*-\s*\$?\d+', r'\w+\s+to\s+\w+', r'\bfrom\s+\w+\s+to\s+\w+',
            r'\bbetween\s+\w+\s+and\s+\w+',
            
            # Size patterns
            r'\b[xX]*[smlSML]\b', r'\bsmall\b', r'\bmedium\b', r'\blarge\b',
            r'\bextra\s+small\b', r'\bextra\s+large\b',
            
            # Single word responses that are clearly follow-ups
            r'^\s*(relaxed|tailored|fitted|loose|tight)\s*$',
            r'^\s*(cotton|silk|linen|wool|polyester)\s*$',
            r'^\s*(work|party|casual|formal|date|brunch)\s*$'
        ]
        
        for pattern in guided_response_patterns:
            if re.search(pattern, query_lower):
                return True
        
        # Check for "no preference" type responses (these are follow-ups, not new searches)
        no_preference_phrases = [
            "no preference", "no preferences", "doesn't matter", "don't care", 
            "anything works", "anything", "any", "whatever", "doesn't matter to me",
            "i don't care", "no specific preference", "not picky", "flexible",
            "no budget", "no budget range", "no price range", "no price limit",
            "no maximum", "no minimum", "no max", "no min", "any budget", 
            "any price", "budget doesn't matter", "price doesn't matter"
        ]
        if any(phrase in query_lower for phrase in no_preference_phrases):
            return True
        
        # More context-aware "i want" detection - only if it's correcting something specific
        # rather than starting a new search
        if any(phrase in query_lower for phrase in ["i want", "i need", "i prefer"]):
            # If it contains product categories, it's likely a new search
            product_categories = ["dress", "top", "skirt", "pants", "coat", "jacket", "shirt", "blouse"]
            if any(category in query_lower for category in product_categories):
                return False  # This is a new product search, not a follow-up
            # If it's "i want it" or "i want that" followed by attributes, it's likely a follow-up
            elif any(phrase in query_lower for phrase in ["i want it", "i want that", "i want them"]) or \
                 any(attr in query_lower for attr in ["size", "budget", "price", "color", "fit", "in red", "in blue", "in black"]):
                return True
                
        return False
    
    def looks_like_product_search(self, query):
        """Detect if a query looks like searching for products rather than a correction/follow-up."""
        query_lower = query.lower().strip()
        
        # First check if it matches our guided response formats - if so, it's NOT a product search
        if self.is_follow_up_or_correction(query):
            return False
        
        # Product search indicators - but only if they're not part of guided responses
        product_indicators = [
            "looking for", "want", "need", "find", "show me", "search",
            "i want a", "i need a", "show me some", "find me",
            "cute", "pretty", "nice", "professional", "casual", "formal",
            "brunch", "work", "party", "date", "wedding", "vacation",
            "comfortable", "trendy", "stylish", "chic", "elegant",
            "something", "anything", "outfit"
        ]
        
        # Check if query contains product search indicators combined with categories
        product_categories = ["dress", "top", "skirt", "pants", "jeans", "shirt", "blouse", "coat", "jacket"]
        has_product_indicator = any(indicator in query_lower for indicator in product_indicators)
        has_product_category = any(category in query_lower for category in product_categories)
        
        # It's a product search if it has both indicators and categories
        if has_product_indicator and has_product_category:
            return True
            
        # If it's a very short query and starts with clear search terms, assume it's a search
        search_starters = ["want", "need", "show", "find", "looking"]
        if len(query_lower.split()) <= 3 and any(query_lower.startswith(starter) for starter in search_starters):
            return True
            
        return False

    # ...existing code...
