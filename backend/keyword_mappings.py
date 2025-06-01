# Comprehensive Keyword to Attribute Mapping System
# This handles vibe-based keywords across all apparel categories

KEYWORD_MAPPINGS = {
    # STYLE VIBES
    "comfy": {"fit": "Relaxed"},
    "comfortable": {"fit": "Relaxed"},
    "cozy": {"fit": "Relaxed", "fabric": ["Cotton", "Modal jersey", "Fleece"]},
    "relaxed": {"fit": "Relaxed"},
    "loose": {"fit": "Relaxed"},
    
    "fitted": {"fit": "Body hugging"},
    "tight": {"fit": "Body hugging"},
    "bodycon": {"fit": "Body hugging"},
    "form-fitting": {"fit": "Body hugging"},
    "clingy": {"fit": "Body hugging"},
    
    "tailored": {"fit": "Tailored"},
    "structured": {"fit": "Tailored"},
    "professional": {"fit": "Tailored", "occasion": "Work"},
    
    # OCCASION KEYWORDS
    "work": {"occasion": "Work"},
    "office": {"occasion": "Work"},
    "business": {"occasion": "Work"},
    "professional": {"occasion": "Work"},
    "corporate": {"occasion": "Work"},
    
    "party": {"occasion": "Party"},
    "date": {"occasion": "Party"},
    "night out": {"occasion": "Party"},
    "clubbing": {"occasion": "Party"},
    "evening": {"occasion": "Party"},
    "formal": {"occasion": "Party"},
    
    "casual": {"occasion": "Casual"},
    "everyday": {"occasion": "Casual"},
    "weekend": {"occasion": "Casual"},
    "lounging": {"occasion": "Casual"},
    
    # FABRIC KEYWORDS
    "silk": {"fabric": "Silk"},
    "silky": {"fabric": "Silk"},
    "luxe": {"fabric": ["Silk", "Satin", "Velvet"]},
    "luxury": {"fabric": ["Silk", "Satin", "Velvet"]},
    "premium": {"fabric": ["Silk", "Satin", "Velvet"]},
    
    "cotton": {"fabric": "Cotton"},
    "breathable": {"fabric": ["Cotton", "Linen", "Tencel"]},
    "natural": {"fabric": ["Cotton", "Linen"]},
    
    "linen": {"fabric": "Linen"},
    
    # SEASONAL KEYWORDS
    "summer": {"fabric": ["Linen", "Cotton"], "sleeve_length": ["Short sleeves", "Sleeveless"], "category": "dress"},
    "summer options": {"fabric": ["Linen", "Cotton"], "sleeve_length": ["Short sleeves", "Sleeveless"]},
    "summer vibes": {"fabric": ["Linen", "Cotton"], "sleeve_length": ["Short sleeves", "Sleeveless"]},
    "summer weather": {"fabric": ["Linen", "Cotton"], "sleeve_length": ["Short sleeves", "Sleeveless"]},
    
    "winter": {"fabric": ["Wool", "Fleece", "Cotton"], "sleeve_length": ["Long sleeves", "Full sleeves"]},
    "winter options": {"fabric": ["Wool", "Fleece", "Cotton"], "sleeve_length": ["Long sleeves", "Full sleeves"]},
    "winter vibes": {"fabric": ["Wool", "Fleece", "Cotton"], "sleeve_length": ["Long sleeves", "Full sleeves"]},
    "winter weather": {"fabric": ["Wool", "Fleece", "Cotton"], "sleeve_length": ["Long sleeves", "Full sleeves"]},
    
    "spring": {"fabric": ["Cotton", "Linen"], "sleeve_length": ["Short sleeves", "Long sleeves"]},
    "spring options": {"fabric": ["Cotton", "Linen"], "sleeve_length": ["Short sleeves", "Long sleeves"]},
    
    "fall": {"fabric": ["Cotton", "Wool"], "sleeve_length": ["Long sleeves", "Full sleeves"]},
    "autumn": {"fabric": ["Cotton", "Wool"], "sleeve_length": ["Long sleeves", "Full sleeves"]},
    
    "satin": {"fabric": "Satin"},
    "shiny": {"fabric": ["Satin", "Silk"]},
    "glossy": {"fabric": ["Satin", "Silk"]},
    
    "stretchy": {"fabric": ["Stretch", "Modal jersey", "Elastane"]},
    "stretch": {"fabric": ["Stretch", "Modal jersey"]},
    "elastic": {"fabric": ["Stretch", "Modal jersey"]},
    
    # COLOR/PRINT KEYWORDS
    "black": {"color_or_print": "Black"},
    "white": {"color_or_print": "White"},
    "navy": {"color_or_print": "Navy"},
    "blue": {"color_or_print": "Blue"},
    "red": {"color_or_print": "Red"},
    "green": {"color_or_print": "Green"},
    "pink": {"color_or_print": "Pink"},
    "yellow": {"color_or_print": "Yellow"},
    "grey": {"color_or_print": "Grey"},
    "gray": {"color_or_print": "Grey"},
    "beige": {"color_or_print": "Beige"},
    "cream": {"color_or_print": "Cream"},
    
    "floral": {"color_or_print": "Floral"},
    "flowers": {"color_or_print": "Floral"},
    "flowery": {"color_or_print": "Floral"},
    "botanical": {"color_or_print": "Floral"},
    
    "striped": {"color_or_print": "Striped"},
    "stripes": {"color_or_print": "Striped"},
    
    "solid": {"color_or_print": "Solid"},
    "plain": {"color_or_print": "Solid"},
    
    "pastel": {"color_or_print": "Pastel"},
    "light": {"color_or_print": "Pastel"},
    "soft": {"color_or_print": "Pastel"},
    
    "bright": {"color_or_print": "Bright"},
    "vibrant": {"color_or_print": "Bright"},
    "bold": {"color_or_print": "Bright"},
    
    # SLEEVE KEYWORDS (for tops/dresses)
    "sleeveless": {"sleeve_length": "Sleeveless"},
    "tank": {"sleeve_length": "Sleeveless"},
    "strapless": {"sleeve_length": "Strapless"},
    
    "short sleeve": {"sleeve_length": "Short sleeves"},
    "short sleeves": {"sleeve_length": "Short sleeves"},
    "tee": {"sleeve_length": "Short sleeves"},
    "t-shirt": {"sleeve_length": "Short sleeves"},
    
    "long sleeve": {"sleeve_length": "Long sleeves"},
    "long sleeves": {"sleeve_length": "Long sleeves"},
    "full sleeve": {"sleeve_length": "Full sleeves"},
    "full sleeves": {"sleeve_length": "Full sleeves"},
    
    # NECKLINE KEYWORDS (for tops/dresses)
    "v-neck": {"neckline": "V-neck"},
    "vneck": {"neckline": "V-neck"},
    "scoop": {"neckline": "Scoop neck"},
    "round neck": {"neckline": "Round neck"},
    "crew": {"neckline": "Round neck"},
    "collar": {"neckline": "Collar"},
    "button-up": {"neckline": "Collar"},
    "halter": {"neckline": "Halter"},
    "off-shoulder": {"neckline": "Off-shoulder"},
    "square": {"neckline": "Square neck"},
    "high neck": {"neckline": "High neck"},
    "turtle": {"neckline": "High neck"},
    "turtleneck": {"neckline": "High neck"},
    
    # LENGTH KEYWORDS (for dresses/skirts)
    "mini": {"length": "Mini"},
    "short": {"length": "Mini"},
    "midi": {"length": "Midi"},
    "knee": {"length": "Midi"},
    "knee-length": {"length": "Midi"},
    "maxi": {"length": "Maxi"},
    "long": {"length": "Maxi"},
    "floor": {"length": "Maxi"},
    "ankle": {"length": "Maxi"},
    
    # PANT TYPE KEYWORDS
    "cargo": {"pant_type": "Cargo", "category": "pants"},
    "cargos": {"pant_type": "Cargo", "category": "pants"},
    "cargo pants": {"pant_type": "Cargo", "category": "pants"},
    "utility": {"pant_type": "Cargo", "category": "pants"},
    
    "jeans": {"pant_type": "Denim", "category": "pants"},
    "denim": {"pant_type": "Denim", "category": "pants"},
    "jean": {"pant_type": "Denim", "category": "pants"},
    
    "wide": {"pant_type": "Wide leg", "category": "pants"},
    "wide leg": {"pant_type": "Wide leg", "category": "pants"},
    "palazzo": {"pant_type": "Wide leg", "category": "pants"},
    
    "straight": {"pant_type": "Straight", "category": "pants"},
    "bootcut": {"pant_type": "Bootcut", "category": "pants"},
    "skinny": {"pant_type": "Skinny", "category": "pants"},
    "slim": {"pant_type": "Slim", "category": "pants"},
    
    "trousers": {"category": "pants"},
    "pants": {"category": "pants"},
    
    # CATEGORY KEYWORDS
    "top": {"category": "top"},
    "tops": {"category": "top"},
    "shirt": {"category": "top"},
    "blouse": {"category": "top"},
    "tee": {"category": "top"},
    "tank": {"category": "top"},
    
    "dress": {"category": "dress"},
    "dresses": {"category": "dress"},
    "gown": {"category": "dress"},
    
    "skirt": {"category": "skirt"},
    "skirts": {"category": "skirt"},
    
    # STYLE COMBINATIONS
    "boho": {"fit": "Relaxed", "color_or_print": "Floral", "fabric": ["Linen", "Cotton"]},
    "bohemian": {"fit": "Relaxed", "color_or_print": "Floral", "fabric": ["Linen", "Cotton"]},
    
    "minimalist": {"color_or_print": "Solid", "fit": "Tailored"},
    "classic": {"color_or_print": "Solid", "fit": "Tailored"},
    "timeless": {"color_or_print": "Solid", "fit": "Tailored"},
    
    # SOPHISTICATION/CLASSY KEYWORDS
    "classy": {"fit": "Tailored", "fabric": ["Silk", "Satin", "Wool"], "color_or_print": ["Solid", "Black", "Navy"], "occasion": "Work"},
    "sophisticated": {"fit": "Tailored", "fabric": ["Silk", "Satin", "Wool"], "color_or_print": ["Solid", "Black", "Navy"]},
    "elegant": {"fit": "Tailored", "fabric": ["Silk", "Satin"], "color_or_print": ["Solid", "Black", "Navy"]},
    "refined": {"fit": "Tailored", "fabric": ["Silk", "Wool"], "color_or_print": "Solid"},
    "polished": {"fit": "Tailored", "fabric": ["Silk", "Satin"], "color_or_print": "Solid"},
    "upscale": {"fit": "Tailored", "fabric": ["Silk", "Satin", "Wool"], "color_or_print": "Solid"},
    
    "edgy": {"color_or_print": "Black", "fit": "Body hugging"},
    "goth": {"color_or_print": "Black", "fit": "Body hugging"},
    "punk": {"color_or_print": "Black", "fit": "Body hugging"},
    
    "feminine": {"color_or_print": ["Floral", "Pastel pink"], "fit": "Body hugging"},
    "girly": {"color_or_print": ["Floral", "Pastel pink"], "fit": "Body hugging"},
    
    "sporty": {"fabric": ["Cotton", "Modal jersey"], "fit": "Relaxed"},
    "athletic": {"fabric": ["Cotton", "Modal jersey"], "fit": "Relaxed"},
    "gym": {"fabric": ["Cotton", "Modal jersey"], "fit": "Relaxed"},
}

def get_keyword_attributes(keyword):
    """
    Get attribute mappings for a given keyword.
    Returns empty dict if keyword not found.
    """
    return KEYWORD_MAPPINGS.get(keyword.lower(), {})

def search_keywords(query):
    """
    Search for all matching keywords in a query string.
    Returns combined attribute mappings with deduplication.
    Uses word boundaries to avoid partial matches.
    """
    import re
    query_lower = query.lower()
    combined_attributes = {}
    
    for keyword, attributes in KEYWORD_MAPPINGS.items():
        # Use word boundaries for single words, substring match for phrases
        if ' ' in keyword:
            # For phrases, use substring matching
            if keyword in query_lower:
                for attr, value in attributes.items():
                    if attr in combined_attributes:
                        # If attribute already exists, combine values with deduplication
                        existing = combined_attributes[attr]
                        if isinstance(existing, list):
                            if isinstance(value, list):
                                # Combine lists and deduplicate
                                combined_list = existing + value
                                combined_attributes[attr] = list(dict.fromkeys(combined_list))  # Preserves order
                            else:
                                if value not in existing:
                                    combined_attributes[attr].append(value)
                        else:
                            # Existing is not a list, convert to list
                            if isinstance(value, list):
                                combined_attributes[attr] = [existing] + value
                                combined_attributes[attr] = list(dict.fromkeys(combined_attributes[attr]))
                            else:
                                if existing != value:
                                    combined_attributes[attr] = [existing, value]
                    else:
                        combined_attributes[attr] = value
        else:
            # For single words, use word boundaries to avoid partial matches
            pattern = r'\b' + re.escape(keyword) + r'\b'
            if re.search(pattern, query_lower):
                for attr, value in attributes.items():
                    if attr in combined_attributes:
                        # If attribute already exists, combine values with deduplication
                        existing = combined_attributes[attr]
                        if isinstance(existing, list):
                            if isinstance(value, list):
                                # Combine lists and deduplicate
                                combined_list = existing + value
                                combined_attributes[attr] = list(dict.fromkeys(combined_list))  # Preserves order
                            else:
                                if value not in existing:
                                    combined_attributes[attr].append(value)
                        else:
                            # Existing is not a list, convert to list
                            if isinstance(value, list):
                                combined_attributes[attr] = [existing] + value
                                combined_attributes[attr] = list(dict.fromkeys(combined_attributes[attr]))
                            else:
                                if existing != value:
                                    combined_attributes[attr] = [existing, value]
                    else:
                        combined_attributes[attr] = value
    
    return combined_attributes
