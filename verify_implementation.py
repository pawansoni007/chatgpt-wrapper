#!/usr/bin/env python3
"""
Syntax Check Script - Verify Phase 3B Implementation
Checks for import errors and basic syntax issues before running the server
"""

import sys
import traceback

def check_imports():
    """Check if all imports work correctly"""
    print("🔍 Checking Python imports...")
    
    try:
        # Test basic imports
        import numpy as np
        import json
        import redis
        import cohere
        from datetime import datetime
        print("✅ Basic dependencies imported successfully")
        
        # Test our custom modules  
        sys.path.append('/Users/pawansoni/chat-wrapper/src')
        
        print("🧵 Testing thread detection system import...")
        from thread_detection_system import ThreadAwareConversationManager
        print("✅ Thread detection system imported successfully")
        
        print("🧠 Testing conversation memory import...")
        from conversation_memory import EnhancedConversationManager
        print("✅ Conversation memory imported successfully")
        
        print("📱 Testing main application import...")
        # Don't actually import main as it will try to start the server
        # Just check the file exists and has basic syntax
        main_path = '/Users/pawansoni/chat-wrapper/src/main.py'
        with open(main_path, 'r') as f:
            content = f.read()
        
        # Basic syntax check by compiling
        compile(content, main_path, 'exec')
        print("✅ Main application syntax is valid")
        
        return True
        
    except ImportError as e:
        print(f"❌ Import error: {e}")
        print("Make sure all dependencies are installed:")
        print("pip install fastapi redis cohere numpy tiktoken cerebras-cloud-sdk")
        return False
    except SyntaxError as e:
        print(f"❌ Syntax error: {e}")
        print("There's a syntax error in the code that needs to be fixed")
        return False
    except Exception as e:
        print(f"❌ Unexpected error: {e}")
        traceback.print_exc()
        return False

def check_redis_connection():
    """Check if Redis is accessible"""
    print("\n🔍 Checking Redis connection...")
    try:
        import redis
        redis_client = redis.Redis(host='localhost', port=6379, db=0, decode_responses=True)
        redis_client.ping()
        print("✅ Redis connection successful")
        return True
    except Exception as e:
        print(f"❌ Redis connection failed: {e}")
        print("Make sure Redis is running: brew services start redis")
        return False

def check_file_structure():
    """Check if all required files exist"""
    print("\n🔍 Checking file structure...")
    
    required_files = [
        '/Users/pawansoni/chat-wrapper/src/main.py',
        '/Users/pawansoni/chat-wrapper/src/thread_detection_system.py',
        '/Users/pawansoni/chat-wrapper/src/conversation_memory.py',
        '/Users/pawansoni/chat-wrapper/models/chat_models.py',
        '/Users/pawansoni/chat-wrapper/tests/test_thread_detection.py',
        '/Users/pawansoni/chat-wrapper/tests/test_basic.py',
        '/Users/pawansoni/chat-wrapper/tests/manual_test.sh'
    ]
    
    all_exist = True
    for file_path in required_files:
        try:
            with open(file_path, 'r') as f:
                pass
            print(f"✅ {file_path.split('/')[-1]}")
        except FileNotFoundError:
            print(f"❌ Missing: {file_path}")
            all_exist = False
    
    return all_exist

def main():
    """Run all checks"""
    print("🧪 PHASE 3B IMPLEMENTATION VERIFICATION")
    print("="*50)
    
    checks = [
        ("File Structure", check_file_structure),
        ("Python Imports", check_imports),
        ("Redis Connection", check_redis_connection)
    ]
    
    passed = 0
    for check_name, check_func in checks:
        try:
            if check_func():
                passed += 1
            else:
                print(f"\n⚠️  {check_name} check failed")
        except Exception as e:
            print(f"\n❌ {check_name} check error: {e}")
    
    print(f"\n🎯 VERIFICATION RESULTS: {passed}/{len(checks)} checks passed")
    
    if passed == len(checks):
        print("\n🎉 PHASE 3B IMPLEMENTATION READY!")
        print("✅ All syntax and dependencies are correct")
        print("🚀 You can now start the server:")
        print("   cd /Users/pawansoni/chat-wrapper")
        print("   python -m uvicorn src.main:app --reload")
        print("\n🧪 Then run tests:")
        print("   python tests/test_basic.py")
        return True
    else:
        print(f"\n❌ IMPLEMENTATION NEEDS ATTENTION")
        print("Please fix the failing checks before starting the server")
        return False

if __name__ == "__main__":
    main()
