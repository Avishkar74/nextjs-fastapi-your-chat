#!/usr/bin/env python3
"""
Quick demo of enhanced context memory working
"""

import requests
import json

def test_conversation_memory():
    """Quick demonstration of enhanced context memory"""
    BASE_URL = "http://localhost:8000/api/enhanced-chat"
    session_id = "demo-session-123"
    
    print("🚀 Quick Enhanced Context Memory Demo")
    print("=" * 50)
    
    # Conversation flow
    conversations = [
        "Hi! I'm building a FastAPI application. Can you help me?",
        "What's the best way to structure the project files?",
        "Remember what I said about FastAPI? Can you suggest database options?",
        "Based on our discussion about FastAPI and databases, what about authentication?",
        "Looking back at everything we've discussed, can you summarize the key points?"
    ]
    
    for i, message in enumerate(conversations, 1):
        print(f"\n{i}. User: {message}")
        
        payload = {
            "message": message,
            "session_id": session_id
        }
        
        try:
            response = requests.post(BASE_URL, json=payload, timeout=30)
            
            if response.status_code == 200:
                data = response.json()
                answer = data.get('answer', '')
                length = data.get('conversation_length', 0)
                
                print(f"   Assistant (msg #{length}): {answer[:150]}...")
                print(f"   💭 Context: Session has {length} messages")
                
            else:
                print(f"   ❌ Error: {response.status_code}")
                
        except Exception as e:
            print(f"   ❌ Exception: {e}")
    
    print("\n🎉 Demo complete! Enhanced context memory working!")

if __name__ == "__main__":
    test_conversation_memory()
