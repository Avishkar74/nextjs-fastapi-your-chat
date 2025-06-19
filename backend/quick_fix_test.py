#!/usr/bin/env python3
"""
Quick fix verification and server restart helper
"""

import requests
import json

def test_enhanced_endpoint():
    """Test the enhanced endpoint to see if it's working"""
    url = "http://localhost:8000/api/enhanced-chat"
    payload = {
        "message": "Simple test after fix",
        "session_id": "fix-test"
    }
    
    try:
        print("🔍 Testing enhanced endpoint after fix...")
        response = requests.post(url, json=payload, timeout=10)
        
        if response.status_code == 200:
            data = response.json()
            print("✅ SUCCESS! Enhanced endpoint is working!")
            print(f"Session ID: {data.get('session_id')}")
            print(f"Conversation Length: {data.get('conversation_length')}")
            print(f"Sources type: {type(data.get('sources'))}")
            print(f"Sources: {data.get('sources')[:1] if data.get('sources') else 'None'}")
            return True
        else:
            print(f"❌ Error {response.status_code}: {response.text}")
            return False
            
    except requests.exceptions.ConnectionError:
        print("❌ Server not running. Please restart the server:")
        print("   1. Stop the current server (Ctrl+C)")
        print("   2. Run: python main.py")
        print("   3. Run this test again")
        return False
    except Exception as e:
        print(f"❌ Error: {e}")
        return False

if __name__ == "__main__":
    print("🚀 Enhanced RAG System Fix Verification")
    print("=" * 50)
    
    # Test the endpoint
    success = test_enhanced_endpoint()
    
    if success:
        print("\n🎉 The fix is working! Enhanced RAG system is ready!")
    else:
        print("\n❌ Fix not working. Server needs restart or there's still an issue.")
        print("\nTo restart the server:")
        print("1. In the server terminal, press Ctrl+C")
        print("2. Run: python main.py")
        print("3. Run this test again: python quick_fix_test.py")
