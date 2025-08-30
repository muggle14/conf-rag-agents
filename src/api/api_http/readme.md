# Ask a new question
curl -X POST $URL/api/query?code=$KEY -d '{"question":"Reset VPN token?"}'

# Submit a clarification
curl -X POST $URL/api/clarify/{conversation_id}?code=$KEY \
     -d '{"clarification":"I mean the mobile soft-token"}'

# Get conversation history
curl -X GET $URL/api/conversation/{conversation_id}?code=$KEY

# Similar queries
curl -X GET "$URL/api/search/similar?query=VPN&limit=3&code=$KEY"
