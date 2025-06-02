#!/usr/bin/env python3
"""
Test suite for Vibe Shopping Agent core functionality.
"""

import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'backend'))

from backend.agent import VibeShoppingAgent

def test_catalog_compatibility():
    """Test catalog compatibility checking for unavailable items."""
    agent = VibeShoppingAgent()
    
    # Test unavailable item
    response = agent.process_query('I want a coat')
    assert response['response_type'] == 'message'
    assert 'not available' in response['message'].lower() or 'do not carry' in response['message'].lower()
    print("✅ Catalog compatibility test passed")

def test_no_preference_handling():
    """Test that 'no preference' responses don't reset conversation."""
    agent = VibeShoppingAgent()
    
    # Start conversation
    agent.process_query('I want a dress')
    
    # Use no preference
    response = agent.process_query('no preference')
    
    # Should not reset conversation
    assert response['response_type'] != 'message' or 'reset' not in response.get('message', '').lower()
    print("✅ No preference handling test passed")

def test_guided_responses():
    """Test that guided response formats are handled correctly."""
    agent = VibeShoppingAgent()
    
    # Start conversation
    agent.process_query('I need pants')
    
    # Test guided response formats
    test_responses = ['under $100', 'M to L', 'relaxed']
    
    for response_text in test_responses:
        response = agent.process_query(response_text)
        # Should not reset conversation
        assert response['response_type'] != 'message' or 'reset' not in response.get('message', '').lower()
    
    print("✅ Guided response format test passed")

def test_follow_up_questions():
    """Test that follow-up questions include helpful examples."""
    agent = VibeShoppingAgent()
    
    response = agent.process_query('Show me dresses')
    
    if response['response_type'] == 'follow_up':
        question = response['questions'][0]
        assert '(e.g.,' in question  # Should have examples
        print("✅ Follow-up question format test passed")

if __name__ == "__main__":
    print("Running Vibe Shopping Agent tests...")
    
    test_catalog_compatibility()
    test_no_preference_handling()
    test_guided_responses()
    test_follow_up_questions()
    
    print("\n🎉 All tests passed!")
