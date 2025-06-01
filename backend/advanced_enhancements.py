import json
import pandas as pd
from datetime import datetime
from collections import defaultdict
import re
from difflib import SequenceMatcher

class AdvancedVibeEngine:
    """Advanced vibe understanding with sophisticated style combinations and seasonal mappings"""
    
    def __init__(self):
        self.seasonal_mappings = {
            'spring': {
                'colors': ['pastel', 'light pink', 'mint green', 'lavender', 'cream'],
                'fabrics': ['cotton', 'linen', 'chambray'],
                'styles': ['flowy', 'light', 'breathable', 'fresh']
            },
            'summer': {
                'colors': ['bright', 'coral', 'turquoise', 'white', 'yellow'],
                'fabrics': ['linen', 'cotton', 'rayon', 'chiffon'],
                'styles': ['breezy', 'sleeveless', 'shorts', 'sundress']
            },
            'fall': {
                'colors': ['burgundy', 'mustard', 'rust', 'olive', 'brown'],
                'fabrics': ['wool', 'knit', 'flannel', 'corduroy'],
                'styles': ['layered', 'cozy', 'warm', 'textured']
            },
            'winter': {
                'colors': ['deep', 'navy', 'black', 'grey', 'jewel tones'],
                'fabrics': ['wool', 'cashmere', 'fleece', 'thick cotton'],
                'styles': ['warm', 'layered', 'long sleeve', 'heavy']
            }
        }
        
        self.style_combinations = {
            'boho chic': {
                'attributes': {'fit': 'relaxed', 'fabric': 'flowy'},
                'colors': ['earth tones', 'floral', 'paisley'],
                'occasions': ['casual', 'festival']
            },
            'minimalist': {
                'attributes': {'fit': 'clean', 'color_or_print': 'solid'},
                'colors': ['black', 'white', 'grey', 'navy'],
                'occasions': ['work', 'casual']
            },
            'romantic': {
                'attributes': {'fabric': 'silk', 'neckline': 'feminine'},
                'colors': ['pink', 'floral', 'lace'],
                'occasions': ['date', 'dinner']
            },
            'edgy': {
                'attributes': {'color_or_print': 'black', 'fit': 'fitted'},
                'colors': ['black', 'leather', 'metallic'],
                'occasions': ['night out', 'party']
            },
            'preppy': {
                'attributes': {'fit': 'tailored'},
                'colors': ['navy', 'stripes', 'plaid'],
                'occasions': ['work', 'brunch']
            }
        }
    
    def get_seasonal_attributes(self):
        """Get style attributes based on current season"""
        month = datetime.now().month
        if month in [3, 4, 5]:
            season = 'spring'
        elif month in [6, 7, 8]:
            season = 'summer'
        elif month in [9, 10, 11]:
            season = 'fall'
        else:
            season = 'winter'
        
        return self.seasonal_mappings.get(season, {})
    
    def analyze_style_combination(self, query):
        """Analyze query for sophisticated style combinations"""
        query_lower = query.lower()
        detected_styles = {}
        
        for style, attributes in self.style_combinations.items():
            if style in query_lower or any(word in query_lower for word in style.split()):
                detected_styles[style] = attributes
        
        return detected_styles

class SmartFiltering:
    """Enhanced filtering with fuzzy matching and intelligent recommendations"""
    
    def __init__(self):
        self.color_families = {
            'neutrals': ['black', 'white', 'grey', 'beige', 'cream', 'tan'],
            'warm': ['red', 'orange', 'yellow', 'pink', 'coral', 'burgundy'],
            'cool': ['blue', 'green', 'purple', 'navy', 'teal', 'mint'],
            'earth': ['brown', 'olive', 'rust', 'mustard', 'khaki']
        }
        
        self.price_intelligence = {
            'budget': {'max': 50, 'keywords': ['affordable', 'cheap', 'budget']},
            'mid-range': {'min': 50, 'max': 150, 'keywords': ['reasonable', 'mid-range']},
            'premium': {'min': 150, 'keywords': ['luxury', 'high-end', 'premium']},
            'splurge': {'min': 300, 'keywords': ['designer', 'splurge', 'investment']}
        }
    
    def fuzzy_match_attributes(self, user_input, available_options, threshold=0.6):
        """Fuzzy matching for attribute values"""
        matches = []
        user_input_lower = user_input.lower()
        
        for option in available_options:
            if isinstance(option, str):
                similarity = SequenceMatcher(None, user_input_lower, option.lower()).ratio()
                if similarity >= threshold:
                    matches.append((option, similarity))
        
        return sorted(matches, key=lambda x: x[1], reverse=True)
    
    def match_color_palette(self, user_color, df):
        """Match colors within the same color family"""
        user_color_lower = user_color.lower()
        
        # Find color family
        target_family = None
        for family, colors in self.color_families.items():
            if any(color in user_color_lower for color in colors):
                target_family = family
                break
        
        if target_family:
            family_colors = self.color_families[target_family]
            # Filter products that match any color in the same family
            mask = df['color_or_print'].str.lower().str.contains('|'.join(family_colors), na=False)
            return df[mask]
        
        return df
    
    def intelligent_price_filtering(self, price_hint, df):
        """Smart price filtering based on budget hints"""
        price_hint_lower = price_hint.lower()
        
        for category, info in self.price_intelligence.items():
            if any(keyword in price_hint_lower for keyword in info['keywords']):
                conditions = []
                if 'min' in info:
                    conditions.append(df['price'] >= info['min'])
                if 'max' in info:
                    conditions.append(df['price'] <= info['max'])
                
                if conditions:
                    combined_condition = conditions[0]
                    for condition in conditions[1:]:
                        combined_condition = combined_condition & condition
                    return df[combined_condition]
        
        return df

class ConversationFlowManager:
    """Enhanced conversation flow with memory and learning"""
    
    def __init__(self):
        self.user_preferences = defaultdict(dict)
        self.conversation_history = []
        self.size_learning = {}
        self.style_profiles = {}
    
    def remember_preference(self, user_id, attribute, value):
        """Remember user preferences across sessions"""
        self.user_preferences[user_id][attribute] = value
    
    def get_learned_size(self, user_id, category):
        """Get previously learned size preferences"""
        return self.size_learning.get(user_id, {}).get(category)
    
    def learn_size_preference(self, user_id, category, size):
        """Learn user's size preferences for different categories"""
        if user_id not in self.size_learning:
            self.size_learning[user_id] = {}
        self.size_learning[user_id][category] = size
    
    def build_style_profile(self, user_id, purchase_history):
        """Build user style profile from purchase history"""
        style_counts = defaultdict(int)
        
        for purchase in purchase_history:
            for attr, value in purchase.items():
                if attr in ['fit', 'fabric', 'color_or_print', 'occasion']:
                    style_counts[f"{attr}:{value}"] += 1
        
        # Top preferences become the style profile
        sorted_prefs = sorted(style_counts.items(), key=lambda x: x[1], reverse=True)
        self.style_profiles[user_id] = dict(sorted_prefs[:10])
    
    def suggest_smart_questions(self, missing_attributes, user_profile=None):
        """Generate contextually relevant follow-up questions"""
        questions = []
        
        # Prioritize questions based on user profile and importance
        priority_map = {
            'category': 3,
            'occasion': 2,
            'fit': 2,
            'available_sizes': 1,
            'price': 1
        }
        
        sorted_missing = sorted(missing_attributes, 
                              key=lambda x: priority_map.get(x, 0), reverse=True)
        
        question_templates = {
            'category': "What type of clothing are you looking for?",
            'occasion': "What occasion will you be wearing this for?",
            'fit': "How would you like it to fit?",
            'available_sizes': "What size do you need?",
            'price': "What's your budget range?",
            'color_or_print': "Any color preferences?",
            'fabric': "Any fabric preferences?"
        }
        
        for attr in sorted_missing[:2]:  # Limit to 2 questions
            if attr in question_templates:
                questions.append(question_templates[attr])
        
        return questions

class AdvancedRecommendationEngine:
    """Advanced recommendations with outfit completion and trend awareness"""
    
    def __init__(self, df):
        self.df = df
        self.outfit_rules = {
            'top': ['pants', 'skirt'],
            'dress': [],  # Complete outfit
            'pants': ['top'],
            'skirt': ['top']
        }
        
        self.trending_items = self.identify_trending_items()
    
    def identify_trending_items(self):
        """Identify trending items (simplified - could use external trend data)"""
        # For demo, assume certain keywords indicate trending items
        trend_keywords = ['oversized', 'cropped', 'high waisted', 'wide leg', 'balloon']
        trending = []
        
        for _, item in self.df.iterrows():
            item_text = f"{item.get('category', '')} {item.get('fabric', '')} {item.get('fit', '')}".lower()
            if any(keyword in item_text for keyword in trend_keywords):
                trending.append(item)
        
        return trending
    
    def suggest_outfit_completion(self, selected_item):
        """Suggest items to complete an outfit"""
        category = selected_item.get('category', '')
        complementary_categories = self.outfit_rules.get(category, [])
        
        suggestions = []
        for comp_category in complementary_categories:
            # Find items in complementary categories
            comp_items = self.df[self.df['category'] == comp_category]
            
            # Filter by compatible attributes (color, style, etc.)
            compatible_items = self.filter_compatible_items(selected_item, comp_items)
            # Clean NaN values before converting to dict
            clean_suggestions = self._clean_nan_values(compatible_items.head(3).to_dict('records'))
            suggestions.extend(clean_suggestions)
        
        return suggestions
    
    def _clean_nan_values(self, items):
        """Clean NaN values from item dictionaries"""
        import math
        clean_items = []
        for item in items:
            clean_item = {}
            for key, value in item.items():
                # Skip NaN values or replace with None/empty string
                if isinstance(value, float) and math.isnan(value):
                    clean_item[key] = None
                else:
                    clean_item[key] = value
            clean_items.append(clean_item)
        return clean_items
    
    def filter_compatible_items(self, base_item, candidate_items):
        """Filter items that would go well together"""
        # Simple compatibility rules
        base_occasion = base_item.get('occasion', '')
        base_color = base_item.get('color_or_print', '')
        
        compatible = candidate_items.copy()
        
        # Filter by occasion compatibility
        if base_occasion and isinstance(base_occasion, str):
            compatible = compatible[
                compatible['occasion'].astype(str).str.contains(base_occasion, na=False) |
                compatible['occasion'].astype(str).str.contains('versatile', na=False)
            ]
        
        # Color compatibility (simplified)
        if base_color and isinstance(base_color, str):
            if 'black' in base_color.lower():
                # Black goes with everything
                pass
            elif 'white' in base_color.lower():
                # White goes with most things
                compatible = compatible[~compatible['color_or_print'].astype(str).str.contains('white', na=False)]
        
        return compatible
    
    def get_customers_also_bought(self, selected_item, limit=5):
        """Simulate 'customers also bought' recommendations"""
        # In a real system, this would use collaborative filtering
        # For demo, find items with similar attributes
        
        similar_items = self.df.copy()
        
        # Score similarity based on shared attributes
        similarity_scores = []
        
        for _, item in similar_items.iterrows():
            if item['id'] == selected_item.get('id'):
                continue
                
            score = 0
            shared_attrs = 0
            
            for attr in ['category', 'occasion', 'fit', 'fabric']:
                if attr in selected_item and attr in item:
                    if selected_item[attr] == item[attr]:
                        score += 1
                    shared_attrs += 1
            
            if shared_attrs > 0:
                similarity_scores.append((item, score / shared_attrs))
        
        # Sort by similarity and return top items
        similarity_scores.sort(key=lambda x: x[1], reverse=True)
        return [item[0] for item in similarity_scores[:limit]]
    
    def get_trending_recommendations(self, user_attributes, limit=5):
        """Get trending items that match user preferences"""
        relevant_trending = []
        
        for item in self.trending_items:
            matches = True
            for attr, value in user_attributes.items():
                if attr in item and item[attr] != value:
                    matches = False
                    break
            
            if matches:
                relevant_trending.append(item)
        
        return relevant_trending[:limit]

class SimilarityRecommendationEngine:
    """Generate similarity-based recommendations for 'Customers Also Bought'"""
    
    def __init__(self, apparel_data):
        self.apparel_data = apparel_data
        self.similarity_cache = {}
    
    def calculate_item_similarity(self, item1, item2, weights=None):
        """Calculate similarity score between two items based on attributes"""
        if weights is None:
            weights = {
                'category': 0.3,
                'fabric': 0.2,
                'color_or_print': 0.15,
                'fit': 0.15,
                'occasion': 0.1,
                'price': 0.1
            }
        
        similarity_score = 0
        
        for attr, weight in weights.items():
            if attr in item1 and attr in item2:
                val1 = str(item1[attr]).lower() if pd.notna(item1[attr]) else ""
                val2 = str(item2[attr]).lower() if pd.notna(item2[attr]) else ""
                
                if attr == 'price':
                    # Price similarity based on range
                    try:
                        price1 = float(val1) if val1 else 0
                        price2 = float(val2) if val2 else 0
                        if price1 > 0 and price2 > 0:
                            price_diff = abs(price1 - price2)
                            max_price = max(price1, price2)
                            attr_similarity = 1 - (price_diff / max_price)
                        else:
                            attr_similarity = 0
                    except:
                        attr_similarity = 0
                else:
                    # Text similarity
                    if val1 == val2:
                        attr_similarity = 1.0
                    elif val1 and val2:
                        # Check for partial matches
                        if val1 in val2 or val2 in val1:
                            attr_similarity = 0.7
                        else:
                            attr_similarity = 0
                    else:
                        attr_similarity = 0
                
                similarity_score += attr_similarity * weight
        
        return similarity_score
    
    def get_customers_also_bought(self, item, num_recommendations=3):
        """Get similar items that 'customers also bought'"""
        item_id = item.get('id', '')
        
        # Check cache first
        cache_key = f"{item_id}_{num_recommendations}"
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        similarities = []
        
        for _, other_item in self.apparel_data.iterrows():
            if other_item.get('id') != item_id:  # Don't recommend the same item
                similarity = self.calculate_item_similarity(item, other_item)
                if similarity > 0.3:  # Minimum similarity threshold
                    similarities.append({
                        'item': other_item,
                        'similarity': similarity
                    })
        
        # Sort by similarity and get top recommendations
        similarities.sort(key=lambda x: x['similarity'], reverse=True)
        
        recommendations = []
        for sim in similarities[:num_recommendations]:
            rec_item = sim['item']
            recommendations.append({
                'id': rec_item.get('id', ''),
                'name': rec_item.get('name', ''),
                'price': rec_item.get('price', 0),
                'similarity_score': sim['similarity']
            })
        
        # Cache the result
        self.similarity_cache[cache_key] = recommendations
        
        return recommendations

# Integration helper functions
def enhance_agent_with_advanced_features(agent):
    """Add advanced features to existing VibeShoppingAgent"""
    agent.vibe_engine = AdvancedVibeEngine()
    agent.smart_filtering = SmartFiltering()
    agent.conversation_manager = ConversationFlowManager()
    agent.recommendation_engine = AdvancedRecommendationEngine(agent.apparel_data)
    
    return agent

def apply_advanced_filtering(df, query, smart_filtering):
    """Apply advanced filtering techniques"""
    # If df is empty or very small, return as is - don't apply additional filters
    if len(df) <= 2:
        print(f"DEBUG: Skipping advanced filtering, only {len(df)} items available")
        return df
    
    # Create a copy for filtering
    filtered_df = df.copy()
    original_count = len(filtered_df)
    
    # Apply fuzzy matching for fabric
    if 'cotton' in query.lower() and 'fabric' in filtered_df.columns:
        try:
            fabric_matches = smart_filtering.fuzzy_match_attributes(query, filtered_df['fabric'].unique())
            if fabric_matches:
                best_match = fabric_matches[0][0]
                temp_df = filtered_df[filtered_df['fabric'].str.contains(best_match, na=False)]
                # Only apply if we don't lose all items
                if len(temp_df) > 0:
                    filtered_df = temp_df
        except Exception as e:
            print(f"Error in fabric filtering: {e}")
    
    # Apply color palette matching
    color_keywords = ['red', 'blue', 'green', 'black', 'white', 'pink', 'purple', 'yellow']
    for color in color_keywords:
        if color in query.lower():
            try:
                temp_df = smart_filtering.match_color_palette(color, filtered_df)
                # Only apply if we don't lose all items
                if len(temp_df) > 0:
                    filtered_df = temp_df
                break
            except Exception as e:
                print(f"Error in color filtering: {e}")
    
    # Apply intelligent price filtering
    budget_keywords = ['cheap', 'affordable', 'budget', 'expensive', 'luxury', 'premium']
    for keyword in budget_keywords:
        if keyword in query.lower():
            try:
                temp_df = smart_filtering.intelligent_price_filtering(keyword, filtered_df)
                # Only apply if we don't lose all items
                if len(temp_df) > 0:
                    filtered_df = temp_df
                break
            except Exception as e:
                print(f"Error in price filtering: {e}")
    
    # If we filtered out all items, return the original dataframe
    if len(filtered_df) == 0:
        print(f"DEBUG: Advanced filtering removed all items, returning original {original_count} items")
        return df
    
    print(f"DEBUG: Advanced filtering: {original_count} -> {len(filtered_df)} items")
    return filtered_df

def generate_enhanced_recommendations(agent, filtered_df, user_attributes):
    """Generate recommendations with advanced features"""
    recommendations = []
    
    # Safety check - if filtered_df is empty, return empty list
    if filtered_df is None or len(filtered_df) == 0:
        print("DEBUG: generate_enhanced_recommendations received empty filtered_df")
        return []
    
    # Ensure better variety by sampling from different price ranges and styles
    total_items = len(filtered_df)
    print(f"DEBUG: generate_enhanced_recommendations received {total_items} items")
    
    if total_items <= 7:
        # If we have 7 or fewer items, show them all
        selected_items = filtered_df
    else:
        # Smart sampling to ensure variety
        # Take first 3 items (likely most relevant)
        top_items = filtered_df.head(3)
        
        # Take some items from the middle and end for variety
        mid_start = total_items // 3
        mid_items = filtered_df.iloc[mid_start:mid_start+2]
        
        # Take some items from the end (including cargo which is last)
        end_items = filtered_df.tail(2)
        
        # Combine them
        selected_items = pd.concat([top_items, mid_items, end_items]).drop_duplicates()
        
        # If we still don't have enough variety, fill with remaining items
        if len(selected_items) < 7:
            remaining_indices = set(filtered_df.index) - set(selected_items.index)
            remaining_items = filtered_df.loc[list(remaining_indices)]
            additional_items = remaining_items.head(7 - len(selected_items))
            selected_items = pd.concat([selected_items, additional_items])
    
    try:
        for _, item in selected_items.iterrows():
            if len(recommendations) >= 7:  # Limit total recommendations
                break
                
            # Convert to dict with proper handling of potential errors
            try:
                rec = item.to_dict()
            except Exception as e:
                print(f"ERROR converting item to dict: {e}")
                # Create a minimal dict with essential properties
                rec = {
                    'id': getattr(item, 'id', ''),
                    'name': getattr(item, 'name', 'Unknown Item'),
                    'price': getattr(item, 'price', 0)
                }
            
            # Add outfit completion suggestions
            try:
                outfit_suggestions = agent.recommendation_engine.suggest_outfit_completion(rec)
                rec['outfit_suggestions'] = outfit_suggestions
            except Exception as e:
                print(f"ERROR adding outfit suggestions: {e}")
                rec['outfit_suggestions'] = []
            
            recommendations.append(rec)
    except Exception as e:
        print(f"ERROR in recommendation generation: {e}")
        # If all else fails, create simple recommendations from the dataframe
        try:
            for i, (_, row) in enumerate(filtered_df.iterrows()):
                if i >= 7:  # Limit to 7 items
                    break
                recommendations.append({
                    'id': str(row.get('id', i)),
                    'name': row.get('name', f'Item {i}'),
                    'price': row.get('price', 0)
                })
        except Exception as e2:
            print(f"CRITICAL ERROR in fallback recommendation: {e2}")
            # Last resort empty list
            return []
    
    # Add trending items if space allows (reduce to 1 to make room for more variety)
    trending = agent.recommendation_engine.get_trending_recommendations(user_attributes, 1)
    for item in trending:
        if len(recommendations) < 7:  # Limit total recommendations
            item['is_trending'] = True
            recommendations.append(item)
    
    return recommendations
