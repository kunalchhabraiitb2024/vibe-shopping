#!/usr/bin/env python3
"""
Integration test for realistic conversation flows.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.agent import VibeShoppingAgent

def test_complete_shopping_flow():
    """Test a complete shopping conversation from start to recommendations."""
    print("Testing complete shopping conversation flow...")
    
    agent = VibeShoppingAgent()
    
    test_queries = [
        "I want something for work",
        "Medium",
        "under $150",
        "no preference"
    ]
    
    for i, query in enumerate(test_queries, 1):
        print(f"\nStep {i}: '{query}'")
        
        result = agent.process_query(query)
        print(f"Response: {result.get('response_type', 'unknown')}")
        
        if result['response_type'] == 'recommendation':
            print(f"Found {len(result.get('recommendations', []))} items")
            break
    
    print("✅ Complete flow test completed")

def test_price_responses():
    """Test various price response formats."""
    agent = VibeShoppingAgent()
    
    agent.process_query('Show me tops')
    
    price_formats = [
        "under $100",
        "$50-120", 
        "at least $30",
        "no budget"
    ]
    
    for price_format in price_formats:
        response = agent.process_query(price_format)
        # Should handle price format without resetting
        assert 'reset' not in response.get('message', '').lower()
    
    print("✅ Price format handling test passed")

if __name__ == "__main__":
    test_complete_shopping_flow()
    test_price_responses()
