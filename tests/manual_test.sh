#!/bin/bash
# Manual Test Script for Phase 3B Thread Detection
# Run this after starting your API server

echo "🧪 MANUAL THREAD DETECTION TESTING"
echo "=================================="

BASE_URL="http://localhost:8000"
CONV_ID="manual-test-$(date +%s)"

echo "Using conversation ID: $CONV_ID"

# Test 1: API Health
echo -e "\n🔍 Test 1: API Health Check"
curl -s "$BASE_URL/" | jq '.'

# Test 2: React Conversation
echo -e "\n📱 Test 2: Starting React Development Conversation"
echo "Message 1: Create React app..."
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Help me create a React to-do app\", \"conversation_id\": \"$CONV_ID\"}" | \
  jq '.message' | head -c 100

echo -e "\n\nMessage 2: Add features..."
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Add due dates to tasks\", \"conversation_id\": \"$CONV_ID\"}" | \
  jq '.message' | head -c 100

# Test 3: Topic Change - Human Development
echo -e "\n\n👥 Test 3: Switching to Human Development"
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Now tell me about human lifecycle development\", \"conversation_id\": \"$CONV_ID\"}" | \
  jq '.message' | head -c 100

echo -e "\n\nMessage about late adulthood..."
curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"What are the key stages of late adulthood?\", \"conversation_id\": \"$CONV_ID\"}" | \
  jq '.message' | head -c 100

# Test 4: The Critical Test - Apple Question
echo -e "\n\n🍎 Test 4: THE CRITICAL TEST - Apple Question"
echo "This should NOT mention React or human development!"
APPLE_RESPONSE=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"tell me about apple?\", \"conversation_id\": \"$CONV_ID\"}")

echo "Apple response:"
echo "$APPLE_RESPONSE" | jq '.message'

# Check for over-contextualization
echo -e "\n🔍 Checking for over-contextualization..."
if echo "$APPLE_RESPONSE" | grep -qi "react\|to-do\|human\|development\|adulthood"; then
    echo "❌ OVER-CONTEXTUALIZATION DETECTED!"
    echo "Found inappropriate context in apple response"
else
    echo "✅ OVER-CONTEXTUALIZATION FIXED!"
    echo "Apple response is clean"
fi

# Test 5: Context Switching Back to React
echo -e "\n🔄 Test 5: Context Switching - Back to React"
REACT_RETURN=$(curl -s -X POST "$BASE_URL/chat" \
  -H "Content-Type: application/json" \
  -d "{\"message\": \"Back to React - how do I handle errors?\", \"conversation_id\": \"$CONV_ID\"}")

echo "React return response:"
echo "$REACT_RETURN" | jq '.message' | head -c 200

# Test 6: Thread Analysis
echo -e "\n\n🧵 Test 6: Thread Analysis"
echo "Getting thread information..."
curl -s "$BASE_URL/conversations/$CONV_ID/threads" | jq '{
  total_threads: .total_threads,
  active_threads: .active_threads,
  threads: [.threads[] | {topic: .topic, status: .status, message_count: .message_count}]
}'

# Test 7: Debug Information
echo -e "\n🐛 Test 7: Debug Information"
curl -s "$BASE_URL/debug/threads/$CONV_ID" | jq '{
  message_count: .message_count,
  detected_boundaries: .detected_boundaries,
  thread_count: (.threads | length),
  thread_topics: [.threads[] | .topic]
}'

echo -e "\n✅ Manual testing completed!"
echo "Check the responses above to verify thread detection is working correctly."
