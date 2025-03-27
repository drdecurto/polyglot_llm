# polyglot_llm.py - Core implementation of the polyglot persistence system with LLM integration
# Dr. de Curtò ; BARCELONA Supercomputing Center / Universidad Pontificia Comillas / UOC
# Dr. de Zarzà; Universidad de Zaragoza / UOC

import os
import json
from retry_utils import retry_with_backoff
import google.generativeai as genai
from pymongo import MongoClient
import redis
import time
import random
from retry_utils import retry_with_backoff, RetryException

# Mock database classes
class MockCollection:
    def __init__(self, data):
        self.data = data
    
    def find(self, query=None):
        if not query:
            return self.data
        
        results = []
        for item in self.data:
            match = True
            for key, value in query.items():
                if key == "$or":
                    or_match = False
                    for or_clause in value:
                        clause_match = True
                        for or_key, or_value in or_clause.items():
                            if or_key in item:
                                if isinstance(or_value, dict) and "$in" in or_value:
                                    clause_match = any(v in item[or_key] for v in or_value["$in"])
                                else:
                                    clause_match = item[or_key] == or_value
                            else:
                                clause_match = False
                        or_match = or_match or clause_match
                    match = match and or_match
                elif key == "$and":
                    and_match = True
                    for and_clause in value:
                        clause_match = True
                        for and_key, and_value in and_clause.items():
                            # Special handling for nested paths like "experience.company"
                            if "." in and_key:
                                parts = and_key.split(".")
                                if parts[0] == "experience" and len(parts) == 2:
                                    company_match = False
                                    for exp in item.get("experience", []):
                                        if exp.get(parts[1]) == and_value:
                                            company_match = True
                                            break
                                    clause_match = company_match
                                else:
                                    clause_match = False
                            elif and_key == "skills" and isinstance(item.get(and_key), list) and isinstance(and_value, str):
                                clause_match = and_value.lower() in [s.lower() for s in item[and_key]]
                            elif and_key in item:
                                clause_match = item[and_key] == and_value
                            else:
                                clause_match = False
                        and_match = and_match and clause_match
                    match = match and and_match
                elif key in item:
                    if isinstance(value, dict) and "$in" in value:
                        if isinstance(item[key], list):
                            match = match and any(v in item[key] for v in value["$in"])
                        else:
                            match = match and item[key] in value["$in"]
                    # Handle case-insensitive string matching for skills
                    elif key == "skills" and isinstance(value, str) and isinstance(item[key], list):
                        match = match and value.lower() in [s.lower() for s in item[key]]
                    # Handle experience.company matching
                    elif key == "experience.company" and isinstance(item.get("experience"), list):
                        company_match = False
                        for exp in item["experience"]:
                            if exp.get("company") == value:
                                company_match = True
                                break
                        match = match and company_match
                    else:
                        match = match and item[key] == value
                else:
                    match = False
            if match:
                results.append(item)
        return results

class MockMongoDB:
    def __init__(self):
        self.collections = {
            'users': MockCollection([
                {"userId": "user1", "name": "Jane Smith", "skills": ["python", "data science", "machine learning"], 
                 "experience": [{"company": "Google", "title": "Data Scientist"}]},
                {"userId": "user2", "name": "John Doe", "skills": ["java", "cloud computing"], 
                 "experience": [{"company": "Amazon", "title": "Software Engineer"}]},
                {"userId": "user3", "name": "Alice Johnson", "skills": ["machine learning", "NLP"], 
                 "experience": [{"company": "Microsoft", "title": "ML Engineer"}]},
            ])
        }
    
    def __getitem__(self, collection_name):
        if collection_name not in self.collections:
            self.collections[collection_name] = MockCollection([])
        return self.collections[collection_name]

def setup_mongodb():
    print("Using mock MongoDB for testing")
    return MockMongoDB()

class MockRedis:
    def __init__(self):
        self.data = {}
    
    def get(self, key):
        return self.data.get(key)
    
    def set(self, key, value, ex=None):
        self.data[key] = value
        return True
    
    def setex(self, key, time, value):
        self.data[key] = value
        return True

def setup_redis():
    print("Using mock Redis for testing")
    return MockRedis()

class MockNeo4jResult:
    def __init__(self, data):
        self.result_data = data
    
    def data(self):
        # Return the data directly
        return self.result_data

class MockNeo4j:
    def __init__(self):
        # Simple graph representation
        self.nodes = {
            "user1": {"userId": "user1", "name": "Jane Smith"},
            "user2": {"userId": "user2", "name": "John Doe"},
            "user3": {"userId": "user3", "name": "Alice Johnson"},
            "user4": {"userId": "user4", "name": "Bob Wilson"},
            "user5": {"userId": "user5", "name": "Charlie Brown"}
        }
        
        # Edges represented as adjacency list
        self.relationships = {
            "user1": {"CONNECTED_TO": ["user2", "user3"]},
            "user2": {"CONNECTED_TO": ["user1", "user4"]},
            "user3": {"CONNECTED_TO": ["user1", "user5"]},
            "user4": {"CONNECTED_TO": ["user2"]},
            "user5": {"CONNECTED_TO": ["user3"]}
        }
    
    def run_query(self, query, params=None):
        # Very basic query parsing for demo purposes
        if "MATCH" in query and "CONNECTED_TO" in query:
            # This is a simplified connection query
            current_user_id = "user1"  # Assume current user is user1
            connections = []
            
            if current_user_id in self.relationships:
                for user_id in self.relationships[current_user_id].get("CONNECTED_TO", []):
                    if user_id in self.nodes:
                        connections.append(self.nodes[user_id])
            
            return MockNeo4jResult(connections)
        
        # Default empty result
        return MockNeo4jResult([])

# LLM Integration

# Modified setup_gemini function with better error handling
# Updated LLM Integration with enhanced rate limit handling
import os
import json
import time
import random
import google.generativeai as genai
from retry_utils import retry_with_backoff, RetryException

# Modified setup_gemini function with better error handling
def setup_gemini():
    # Get API key from environment variable
    api_key = os.environ.get('GEMINI_API_KEY', 'YOUR_API_KEY_HERE')
    
    if not api_key or api_key == 'YOUR_API_KEY_HERE':
        print("⚠️ Gemini API key not set. Please add it to the environment variables.")
        return None
    
    genai.configure(api_key=api_key)
    
    try:
        # Create the model with appropriate configuration
        generation_config = {
            "temperature": 0.2,  # Lower for more predictable outputs
            "top_p": 0.95,
            "top_k": 40,
            "max_output_tokens": 8192,
        }
        
        # Specific exceptions to handle for API model initialization
        api_exceptions = (
            genai.types.BlockedPromptException, 
            genai.types.StopCandidateException,
            ValueError,
            Exception  # Catch-all as a last resort
        )
        
        # Try different models in order of preference with retry logic
        models_to_try = [
            "gemini-2.0-pro-exp-02-05", 
            "gemini-1.5-pro", 
            "gemini-1.5-flash",
        ]
        
        for model_name in models_to_try:
            try:
                @retry_with_backoff(
                    max_retries=4, 
                    initial_delay=5.0, 
                    backoff_factor=2.5,
                    max_delay=120.0,
                    exceptions_to_retry=api_exceptions
                )
                def initialize_model():
                    return genai.GenerativeModel(
                        model_name=model_name,
                        generation_config=generation_config,
                    )
                
                model = initialize_model()
                print(f"Using {model_name} model")
                print("Gemini API configured successfully!")
                return model
            
            except Exception as e:
                print(f"Error with {model_name}: {e}")
                # Add a delay before trying the next model
                time.sleep(5)
                continue  # Try the next model
                
        # If we get here, all models failed
        print("❌ Could not initialize any Gemini model")
        return None
        
    except Exception as e:
        print(f"Error setting up Gemini: {e}")
        return None

# Modified initialize_llm_session function with retry logic
def initialize_llm_session(model):
    if model is None:
        print("Cannot initialize LLM session: No model available")
        return None
        
    # Sample database schema information
    db_schema = {
        "mongodb": {
            "users": {
                "userId": "string",
                "name": "string",
                "skills": ["string"],
                "experience": [{
                    "title": "string",
                    "company": "string"
                }]
            }
        },
        "neo4j": {
            "relationships": ["CONNECTED_TO", "ENDORSED", "WORKED_WITH"]
        }
    }
    
    system_prompt = f"""
    You are a database query assistant for a social network application.
    You translate natural language queries into structured database queries.
    The system uses a polyglot persistence architecture with:
    1. MongoDB for user profiles and content (schema: {json.dumps(db_schema['mongodb'])})
    2. Neo4j for social relationships (relationships: {', '.join(db_schema['neo4j']['relationships'])})
    3. Redis for caching and frequent lookups
    
    Format your responses as JSON with the following structure:
    {{
        "query_type": "simple"|"complex",
        "databases": ["mongodb", "neo4j", "redis"],
        "mongodb_query": {{...}},
        "neo4j_query": "...",
        "redis_operations": [...],
        "explanation": "..."
    }}
    """
    
    try:
        # Initialize chat session with retry logic
        @retry_with_backoff(max_retries=5, initial_delay=5.0, backoff_factor=2.5, max_delay=120.0, jitter=True)
        def start_chat_with_retry():
            return model.start_chat(
                history=[
                    {"role": "user", "parts": [system_prompt]},
                    {"role": "model", "parts": ["I understand. I'll translate natural language queries into structured database queries for your polyglot persistence architecture, formatting responses as JSON with the specified structure."]}
                ]
            )
        
        chat_session = start_chat_with_retry()
        return chat_session
        
    except Exception as e:
        print(f"Error initializing chat session: {e}")
        return None
    
# Query Processing Classes
# # Modified QueryTranslator class with retry logic
# Modified QueryTranslator class with improved retry logic

class QueryTranslator:
    def __init__(self, llm_session, max_retries=5, initial_delay=5.0):
        self.llm_session = llm_session
        self.max_retries = max_retries
        self.initial_delay = initial_delay
    
    def translate_query(self, natural_language_query):
        if self.llm_session is None:
            print("Cannot translate query: No LLM session available")
            # Return a default query plan
            return {
                "query_type": "simple",
                "databases": ["mongodb"],
                "mongodb_query": {
                    "skills": "data science"
                },
                "explanation": "Default query plan (LLM not available)"
            }
            
        # Send the query to the LLM with retry logic
        try:
            response = self._send_message_with_retry(
                f"Translate this query to appropriate database operations: {natural_language_query}"
            )
            
            # Parse the response
            try:
                # Extract JSON from the response
                response_text = response.text
                # Find JSON in the response (handling potential markdown formatting)
                json_start = response_text.find('{')
                json_end = response_text.rfind('}') + 1
                
                if json_start >= 0 and json_end > json_start:
                    json_str = response_text[json_start:json_end]
                    query_plan = json.loads(json_str)
                    return query_plan
                else:
                    raise ValueError("No JSON found in response")
                    
            except (json.JSONDecodeError, ValueError) as e:
                # Fallback for non-JSON responses
                print(f"Error parsing LLM response: {e}")
                print(f"Raw response: {response.text}")
                return {
                    "query_type": "error",
                    "explanation": "Failed to parse LLM response into a structured query plan",
                    "raw_response": response.text
                }
        except Exception as e:
            print(f"Error getting response from LLM after retries: {e}")
            # Return a simple fallback query plan
            return {
                "query_type": "simple",
                "databases": ["mongodb"],
                "mongodb_query": {
                    "skills": {"$in": ["data science", "machine learning"]}
                },
                "explanation": "Default fallback query plan due to LLM error"
            }
    
    @retry_with_backoff(max_retries=5, initial_delay=5.0, backoff_factor=2.5, max_delay=120.0, jitter=True)
    def _send_message_with_retry(self, message):
        """Send a message to the LLM with retry logic."""
        return self.llm_session.send_message(message)
    
class QueryExecutor:
    def __init__(self, mongodb_client, neo4j_client, redis_client):
        self.mongodb_client = mongodb_client
        self.neo4j_client = neo4j_client
        self.redis_client = redis_client
    
    def execute_query_plan(self, query_plan):
        results = {}
        
        # Execute MongoDB queries
        if "mongodb" in query_plan.get("databases", []):
            mongodb_query = query_plan.get("mongodb_query", {})
            
            # Handle different query formats
            if isinstance(mongodb_query, dict):
                # Check if collection is specified directly or as part of the structure
                if "collection" in mongodb_query:
                    collection_name = mongodb_query.get("collection", "users")
                    query_filter = mongodb_query.get("filter", {})
                elif "users" in mongodb_query:  # Directly specified collection in query
                    collection_name = "users"
                    query_filter = mongodb_query.get("users", {})
                else:
                    collection_name = "users"  # Default
                    query_filter = mongodb_query
            else:
                collection_name = "users"  # Default
                query_filter = {}
            
            collection = self.mongodb_client[collection_name]
            mongo_results = list(collection.find(query_filter))
            
            # Convert ObjectId to string for JSON serialization if needed
            for doc in mongo_results:
                if '_id' in doc and not isinstance(doc['_id'], str):
                    doc['_id'] = str(doc['_id'])
            
            results["mongodb"] = mongo_results
        
        # Execute Neo4j queries
        if "neo4j" in query_plan.get("databases", []):
            neo4j_query = query_plan.get("neo4j_query", "")
            if neo4j_query:
                neo4j_result = self.neo4j_client.run_query(neo4j_query)
                # Get the data directly, not calling data() as a function
                results["neo4j"] = neo4j_result.data()
        
        return results

# Modified ResultSynthesizer class with retry logic
# Modified ResultSynthesizer class with improved retry logic

class ResultSynthesizer:
    def __init__(self, llm_session, max_retries=5, initial_delay=5.0):
        self.llm_session = llm_session
        self.max_retries = max_retries
        self.initial_delay = initial_delay
    
    def synthesize_results(self, query_results, original_query):
        if self.llm_session is None:
            # Create a simple human-readable response without LLM
            mongodb_results = query_results.get("mongodb", [])
            neo4j_results = query_results.get("neo4j", [])
            
            summary = f"Found {len(mongodb_results)} results from MongoDB and {len(neo4j_results)} from Neo4j."
            details = []
            
            for result in mongodb_results:
                name = result.get("name", "Unknown")
                skills = ", ".join(result.get("skills", []))
                experience = [f"{exp.get('title')} at {exp.get('company')}" 
                             for exp in result.get("experience", [])]
                exp_str = "; ".join(experience) if experience else "No experience listed"
                
                details.append(f"- {name}: Skills: {skills}. Experience: {exp_str}")
            
            return summary + "\n\n" + "\n".join(details)
        
        # If LLM is available, use it for better synthesis with retry logic
        try:
            # Prepare the context for the LLM
            context = self._prepare_context(original_query, query_results)
            
            # Send the context to the LLM with retry logic
            response = self._send_message_with_retry(context)
            return response.text
            
        except Exception as e:
            print(f"Error synthesizing results with LLM after retries: {e}")
            # Fallback to simple synthesis
            mongodb_results = query_results.get("mongodb", [])
            result_names = [r.get("name", "Unknown") for r in mongodb_results]
            if result_names:
                return f"I found {len(result_names)} results that match your query: {', '.join(result_names)}"
            else:
                return "I couldn't find any results matching your query."
    
    def _prepare_context(self, original_query, query_results):
        """Prepare the context for the LLM."""
        return f"""
        Original query: {original_query}
        
        Query results:
        {json.dumps(query_results, indent=2)}
        
        Please synthesize these results into a coherent response that addresses the original query.
        Format your response in a conversational manner that would be helpful to a user of a social network.
        DO NOT include the query structure or technical implementation details in your response.
        Focus only on providing a user-friendly answer to the query.
        """
    
    @retry_with_backoff(max_retries=5, initial_delay=5.0, backoff_factor=2.5, max_delay=120.0, jitter=True)
    def _send_message_with_retry(self, message):
        """Send a message to the LLM with retry logic."""
        return self.llm_session.send_message(message)
        
# Main Query Processing Function
def process_natural_language_query(query, llm_session, db_clients):
    # Initialize components
    translator = QueryTranslator(llm_session)
    executor = QueryExecutor(
        db_clients["mongodb"],
        db_clients["neo4j"],
        db_clients["redis"]
    )
    synthesizer = ResultSynthesizer(llm_session)
    
    # Step 1: Translate the natural language query
    print(f"🔄 Translating query: '{query}'")
    query_plan = translator.translate_query(query)
    print(f"📝 Generated query plan:")
    print(json.dumps(query_plan, indent=2))
    
    # Step 2: Execute the query
    print(f"⚙️ Executing query plan...")
    results = executor.execute_query_plan(query_plan)
    print(f"📊 Raw results:")
    print(json.dumps(results, indent=2))
    
    # Step 3: Synthesize the results
    print(f"🔄 Synthesizing results...")
    final_response = synthesizer.synthesize_results(results, query)
    print(f"✅ Final response:")
    print(final_response)
    
    return {
        "original_query": query,
        "query_plan": query_plan,
        "raw_results": results,
        "response": final_response
    }