"""
Dynamic Keyword Mapper for Vibe-Based Shopping Agent

This module implements intelligent keyword extraction from the catalog
and LLM-based semantic mapping of user queries to catalog keywords.
"""

import pandas as pd
import re
from collections import defaultdict, Counter
from typing import Dict, List, Set, Optional
import google.generativeai as genai
import os
from dotenv import load_dotenv
import json

load_dotenv()

class DynamicKeywordMapper:
    def __init__(self, catalog_path: str = "Apparels_shared.xlsx"):
        """Initialize the dynamic keyword mapper with catalog data."""
        self.catalog_path = catalog_path
        self.catalog_keywords = {}
        self.semantic_cache = {}
        
        # Initialize Google Gemini API
        api_key = os.getenv('GOOGLE_API_KEY')
        if api_key:
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel('gemini-1.5-flash')
        else:
            print("Warning: GOOGLE_API_KEY not found. LLM features will not work.")
            self.model = None
        
        # Load and process catalog
        self.load_catalog_keywords()
    
    def load_catalog_keywords(self):
        """Extract unique keywords from catalog for each attribute."""
        try:
            # Read Excel file
            df = pd.read_excel(self.catalog_path)
            
            # Key attributes to extract keywords from
            attributes = [
                'category', 'brand', 'color_or_print', 'fabric', 
                'fit', 'sleeve_length', 'pattern', 'style', 'occasion',
                'neckline', 'length', 'pant_type'
            ]
            
            self.catalog_keywords = {}
            
            for attr in attributes:
                if attr in df.columns:
                    # Get unique non-null values
                    unique_values = df[attr].dropna().unique()
                    
                    # Clean and extract individual keywords
                    keywords = set()
                    for value in unique_values:
                        if isinstance(value, str):
                            # Split on common delimiters and clean
                            parts = re.split(r'[,;/&\-\s]+', value.lower())
                            for part in parts:
                                part = part.strip()
                                if part and len(part) > 1:  # Skip single characters
                                    keywords.add(part)
                    
                    self.catalog_keywords[attr] = sorted(list(keywords))
                    
            print(f"Extracted keywords for {len(self.catalog_keywords)} attributes")
            for attr, keywords in self.catalog_keywords.items():
                print(f"  {attr}: {len(keywords)} keywords")
                
        except Exception as e:
            print(f"Error loading catalog keywords: {e}")
            # Fallback to empty keywords
            self.catalog_keywords = {}
    
    def get_catalog_keywords(self, attribute: str) -> List[str]:
        """Get extracted keywords for a specific attribute."""
        return self.catalog_keywords.get(attribute, [])
    
    def map_user_query_to_keywords(self, user_query: str, last_asked_attribute: Optional[str] = None) -> Dict[str, List[str]]: # MODIFIED: Added last_asked_attribute
        """
        Use LLM to intelligently map user query to catalog keywords.
        
        Args:
            user_query: User's natural language query
            last_asked_attribute: The attribute the user was last asked about (optional)
            
        Returns:
            Dictionary mapping attributes to matched keywords
        """
        # Check cache first
        # Cache key should probably include last_asked_attribute if it significantly changes the outcome
        # For now, keeping it simple. Consider refining cache key if needed.
        query_lower = user_query.lower().strip()
        if query_lower in self.semantic_cache and last_asked_attribute is None: # Simple cache check
            return self.semantic_cache[query_lower]
        
        # Special case for "anything works" - return empty to skip follow-ups
        if any(phrase in query_lower for phrase in ["anything works", "anything's fine", "doesn't matter", "any is fine"]):
            return {"skip_followup": True}
        
        # Extract price information first
        result = self._extract_price_info(user_query)
        
        if not self.model:
            print("No LLM model available, using fallback matching")
            fallback_result = self._fallback_keyword_matching(user_query)
            # Merge price info with fallback result
            result.update(fallback_result)
            return result
        
        try:
            # Prepare catalog keywords for LLM context
            keywords_context = self._format_keywords_for_llm()
            
            context_hint = ""
            if last_asked_attribute:
                context_hint = f"The user was previously asked about the attribute: '{last_asked_attribute}'. If the current query seems to be a direct answer to this (e.g., user says 'comfortable' when asked about 'fit', or 'S' when asked about 'size'), prioritize mapping the query to the '{last_asked_attribute}' attribute."

            prompt = f"""
You are a fashion expert helping to map user queries to specific catalog attributes.

Available catalog keywords by attribute:
{keywords_context}

User query: "{user_query}"

{context_hint}

Your task: Map the user's query to the most relevant catalog keywords. Consider:
- Synonyms (e.g., "elegant" -> "formal", "comfy" -> "casual")
- Style implications (e.g., "professional" -> "tailored fit", "formal occasion")
- Seasonal context (e.g., "winter" -> "wool", "long sleeves", "winter options" -> "wool", "fleece", "long sleeves")
- Color descriptions (e.g., "dark" -> "black", "navy")
- Sophistication terms (e.g., "classy pants" -> "tailored fit", "wool fabric", "solid colors")

Respond with a JSON object where keys are attribute names and values are arrays of matching keywords:

{{
    "category": ["relevant_keywords"],
    "color_or_print": ["relevant_keywords"],
    "fabric": ["relevant_keywords"],
    "fit": ["relevant_keywords"],
    "sleeve_length": ["relevant_keywords"],
    "occasion": ["relevant_keywords"]
}}

Only include attributes that are relevant to the query. Use exact keywords from the catalog. If the query is a direct answer to the `last_asked_attribute`, ensure that attribute is present in your response if a suitable keyword is found.
"""

            response = self.model.generate_content(prompt)
            
            # Parse LLM response
            response_text = response.text.strip()
            
            # Extract JSON from response (handle markdown code blocks)
            if "```json" in response_text:
                json_start = response_text.find("```json") + 7
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            elif "```" in response_text:
                json_start = response_text.find("```") + 3
                json_end = response_text.find("```", json_start)
                response_text = response_text[json_start:json_end].strip()
            
            llm_result = json.loads(response_text)
            
            # Validate keywords exist in catalog
            validated_result = {}
            for attr, keywords in llm_result.items():
                if attr in self.catalog_keywords:
                    valid_keywords = []
                    for keyword in keywords:
                        # Case-insensitive matching
                        catalog_keywords_lower = [k.lower() for k in self.catalog_keywords[attr]]
                        if keyword.lower() in catalog_keywords_lower:
                            # Find original case
                            for orig_keyword in self.catalog_keywords[attr]:
                                if orig_keyword.lower() == keyword.lower():
                                    valid_keywords.append(orig_keyword)
                                    break
                    if valid_keywords:
                        validated_result[attr] = valid_keywords
            
            # Merge price info with validated LLM result
            result.update(validated_result)
            
            # Cache the result
            self.semantic_cache[query_lower] = result
            
            return result
            
        except Exception as e:
            print(f"Error in LLM mapping: {e}")
            # Fallback to simple keyword matching
            fallback_result = self._fallback_keyword_matching(user_query)
            # Merge price info with fallback result
            result.update(fallback_result)
            return result
    
    def _format_keywords_for_llm(self) -> str:
        """Format catalog keywords for LLM context."""
        formatted = []
        for attr, keywords in self.catalog_keywords.items():
            # Limit to most common/relevant keywords to stay within token limits
            limited_keywords = keywords[:20] if len(keywords) > 20 else keywords
            formatted.append(f"{attr}: {', '.join(limited_keywords)}")
        return '\n'.join(formatted)
    
    def _fallback_keyword_matching(self, user_query: str) -> Dict[str, List[str]]:
        """Fallback to simple string matching if LLM fails."""
        result = {}
        query_lower = user_query.lower()
        
        # Special seasonal mappings for fallback
        seasonal_mappings = {
            "winter": {"fabric": ["wool", "fleece"], "sleeve_length": ["long sleeves", "full sleeves"]},
            "summer": {"fabric": ["cotton", "linen"], "sleeve_length": ["short sleeves", "sleeveless"]},
            "spring": {"fabric": ["cotton", "linen"], "sleeve_length": ["short sleeves", "long sleeves"]},
            "fall": {"fabric": ["wool", "cotton"], "sleeve_length": ["long sleeves", "full sleeves"]}
        }
        
        # Check for seasonal terms
        for season, mappings in seasonal_mappings.items():
            if season in query_lower or f"{season} options" in query_lower:
                for attr, values in mappings.items():
                    if attr in self.catalog_keywords:
                        matched = []
                        for value in values:
                            for keyword in self.catalog_keywords[attr]:
                                if value.lower() in keyword.lower():
                                    matched.append(keyword)
                        if matched:
                            result[attr] = matched
        
        # Regular keyword matching
        for attr, keywords in self.catalog_keywords.items():
            matched = []
            for keyword in keywords:
                if keyword.lower() in query_lower:
                    matched.append(keyword)
            if matched and attr not in result:
                result[attr] = matched
        
        return result
    
    def get_semantic_suggestions(self, partial_query: str, max_suggestions: int = 5) -> List[str]:
        """
        Get semantic suggestions for partial queries using catalog keywords.
        
        Args:
            partial_query: Partial user input
            max_suggestions: Maximum number of suggestions to return
            
        Returns:
            List of suggested completions
        """
        suggestions = []
        query_lower = partial_query.lower()
        
        # Find keywords that start with or contain the partial query
        for attr, keywords in self.catalog_keywords.items():
            for keyword in keywords:
                if query_lower in keyword.lower():
                    suggestions.append(keyword)
        
        # Sort by relevance (exact start matches first)
        def sort_key(keyword):
            kw_lower = keyword.lower()
            if kw_lower.startswith(query_lower):
                return (0, len(keyword))  # Exact start, shorter first
            else:
                return (1, len(keyword))  # Contains, shorter first
        
        suggestions.sort(key=sort_key)
        
        return suggestions[:max_suggestions]
    
    def analyze_catalog_trends(self) -> Dict[str, Counter]:
        """Analyze catalog to identify trending attributes."""
        try:
            df = pd.read_excel(self.catalog_path)
            trends = {}
            
            for attr in self.catalog_keywords.keys():
                if attr in df.columns:
                    # Count frequency of each value
                    value_counts = df[attr].value_counts()
                    trends[attr] = Counter(dict(value_counts.head(10)))
            
            return trends
            
        except Exception as e:
            print(f"Error analyzing trends: {e}")
            return {}
    
    def get_attribute_suggestions(self, user_query: str, attribute: str) -> List[str]:
        """
        Get specific suggestions for a given attribute based on user query.
        
        Args:
            user_query: User's query
            attribute: Specific attribute to get suggestions for
            
        Returns:
            List of relevant attribute values
        """
        if attribute not in self.catalog_keywords:
            return []
        
        # Use LLM mapping first
        mapped_keywords = self.map_user_query_to_keywords(user_query)
        if attribute in mapped_keywords:
            return mapped_keywords[attribute]
        
        # Fallback to direct matching
        query_lower = user_query.lower()
        suggestions = []
        
        for keyword in self.catalog_keywords[attribute]:
            if query_lower in keyword.lower() or keyword.lower() in query_lower:
                suggestions.append(keyword)
        
        return suggestions[:5]
    
    def _extract_price_info(self, user_query: str) -> Dict[str, any]:
        """Extract price information from user query."""
        result = {}
        query_lower = user_query.lower()
        
        # Price patterns to detect budget constraints
        price_patterns = [
            (r'under\s+\$?(\d+)', 'price'),  # "under $150" -> price_max: 150
            (r'below\s+\$?(\d+)', 'price'),  # "below $150" -> price_max: 150
            (r'above\s+\$?(\d+)', 'price_min'),  # "above $100" -> price_min: 100
            (r'over\s+\$?(\d+)', 'price_min'),  # "over $100" -> price_min: 100
            (r'between\s+\$?(\d+)\s+and\s+\$?(\d+)', 'range'),  # "between $50 and $150"
            (r'\$(\d+)\s*-\s*\$?(\d+)', 'range'),  # "$50-$150"
            (r'budget\s+is.*?\$?(\d+)', 'price'),  # "budget is $150"
            (r'\$(\d+)', 'price')  # "$150"
        ]
        
        for pattern, price_type in price_patterns:
            price_match = re.search(pattern, query_lower)
            if price_match:
                if price_type == 'price':
                    result["price"] = int(price_match.group(1))
                elif price_type == 'price_min':
                    result["price_min"] = int(price_match.group(1))
                elif price_type == 'range':
                    result["price_min"] = int(price_match.group(1))
                    result["price_max"] = int(price_match.group(2))
                break  # Use first match
        
        return result

# Global instance for easy import
dynamic_mapper = DynamicKeywordMapper()

def get_dynamic_keywords(user_query: str, last_asked_attribute: Optional[str] = None) -> Dict[str, List[str]]: # MODIFIED: Added last_asked_attribute
    """Convenience function to get dynamic keyword mapping."""
    return dynamic_mapper.map_user_query_to_keywords(user_query, last_asked_attribute) # MODIFIED: Pass last_asked_attribute

def get_catalog_keywords(attribute: str) -> List[str]:
    """Convenience function to get catalog keywords for an attribute."""
    return dynamic_mapper.get_catalog_keywords(attribute)

def get_suggestions(partial_query: str, max_suggestions: int = 5) -> List[str]:
    """Convenience function to get semantic suggestions."""
    return dynamic_mapper.get_semantic_suggestions(partial_query, max_suggestions)
