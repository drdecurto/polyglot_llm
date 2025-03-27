# evaluation.py - Evaluation framework for polyglot persistence with LLMs
# Dr. de Curtò ; BARCELONA Supercomputing Center / Universidad Pontificia Comillas / UOC
# Dr. de Zarzà; Universidad de Zaragoza / UOC

import os
import json
import time
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm
import google.generativeai as genai
from pymongo import MongoClient
import redis
from retry_utils import retry_with_backoff, RetryException


from polyglot_llm import (
    setup_mongodb, setup_redis, MockNeo4j, setup_gemini,
    initialize_llm_session, QueryTranslator, QueryExecutor, ResultSynthesizer,
    MockNeo4jResult
)

# Define test queries categorized by complexity
TEST_QUERIES = {
    "simple": [
        "Find people with machine learning skills",
        "Who has worked at Google?",
        "List users with Python skills",
        # other simple queries...
    ],
    "complex": [
        "Find data scientists in my network who worked at Google",
        "Who in my connections has both machine learning and cloud computing skills?",
        # other complex queries...
    ],
    "ambiguous": [
        "Who can help me find a job in AI?",
        "Find experienced professionals in my network",
        # other ambiguous queries...
    ]
}

# Create a synthetic dataset
def generate_synthetic_dataset(num_users=1000, num_connections=5000):
    # Define possible values
    companies = [
        "Google", "Microsoft", "Amazon", "Apple", "Meta", "Netflix", "IBM", 
        "Oracle", "Salesforce", "Adobe", "Twitter", "LinkedIn", "Uber", 
        "Airbnb", "Spotify", "Tesla", "Intel", "AMD", "NVIDIA", "Samsung"
    ] * 10  # Repeat to have 200 companies
    
    skills = [
        "Python", "Java", "JavaScript", "C++", "SQL", "NoSQL", "Docker", "Kubernetes", 
        # other skills...
    ] * 3  # Repeat to have 300 skills
    
    job_titles = [
        "Software Engineer", "Data Scientist", "Machine Learning Engineer", "Data Engineer",
        # other job titles...
    ]
    
    # Generate users
    users = []
    for i in range(1, num_users + 1):
        num_skills = np.random.randint(2, 10)
        num_jobs = np.random.randint(1, 5)
        
        user_skills = np.random.choice(skills, num_skills, replace=False).tolist()
        
        experience = []
        for j in range(num_jobs):
            company = np.random.choice(companies)
            title = np.random.choice(job_titles)
            years = np.random.randint(1, 10)
            experience.append({
                "company": company,
                "title": title,
                "years": years
            })
        
        user = {
            "userId": f"user{i}",
            "name": f"User {i}",
            "email": f"user{i}@example.com",
            "skills": user_skills,
            "experience": experience
        }
        users.append(user)
    
    # Generate connections
    connections = []
    for _ in range(num_connections):
        user1 = np.random.randint(1, num_users + 1)
        user2 = np.random.randint(1, num_users + 1)
        if user1 != user2:  # Avoid self-connections
            connection_type = np.random.choice(["CONNECTED_TO", "ENDORSED", "WORKED_WITH"])
            connections.append({
                "source": f"user{user1}",
                "target": f"user{user2}",
                "type": connection_type
            })
    
    return {
        "users": users,
        "connections": connections
    }

# Mock database classes extended for evaluation
class EvalMockMongoDB:
    def __init__(self, dataset):
        self.collections = {
            'users': EvalMockCollection(dataset["users"])
        }
    
    def __getitem__(self, collection_name):
        if collection_name not in self.collections:
            self.collections[collection_name] = EvalMockCollection([])
        return self.collections[collection_name]

class EvalMockCollection:
    def __init__(self, data):
        self.data = data
    
    def find(self, query=None):
        if not query:
            return self.data
        
        # Implement advanced query handling here
        results = []
        for item in self.data:
            match = True
            
            for key, value in query.items():
                if key == "$or":
                    or_match = False
                    for or_clause in value:
                        clause_match = self._match_clause(item, or_clause)
                        or_match = or_match or clause_match
                    match = match and or_match
                elif key == "$and":
                    and_match = True
                    for and_clause in value:
                        clause_match = self._match_clause(item, and_clause)
                        and_match = and_match and clause_match
                    match = match and and_match
                else:
                    match = match and self._match_field(item, key, value)
            
            if match:
                results.append(item)
        
        return results
    
    def _match_clause(self, item, clause):
        match = True
        for key, value in clause.items():
            match = match and self._match_field(item, key, value)
        return match
    
    def _match_field(self, item, key, value):
        # Handle dot notation (e.g., "experience.company")
        if "." in key:
            parts = key.split(".")
            if parts[0] == "experience" and len(parts) == 2:
                field = parts[1]
                for exp in item.get("experience", []):
                    if field in exp and self._match_value(exp[field], value):
                        return True
                return False
        
        # Handle direct fields
        if key not in item:
            return False
        
        return self._match_value(item[key], value)
    
    def _match_value(self, item_value, query_value):
        # Handle different value types and query operators
        if isinstance(query_value, dict):
            # Handle operators like $in
            if "$in" in query_value:
                if isinstance(item_value, list):
                    return any(v in item_value for v in query_value["$in"])
                else:
                    return item_value in query_value["$in"]
            # Add other operators as needed
            return False
        
        # Handle string matching for skills
        if isinstance(item_value, list) and isinstance(query_value, str):
            return query_value.lower() in [s.lower() for s in item_value]
        
        # Direct comparison
        return item_value == query_value

class EvalMockNeo4j:
    def __init__(self, dataset):
        # Build a node lookup
        self.nodes = {}
        for user in dataset["users"]:
            self.nodes[user["userId"]] = {
                "userId": user["userId"],
                "name": user["name"]
            }
        
        # Build relationship structure
        self.relationships = {}
        for conn in dataset["connections"]:
            source = conn["source"]
            target = conn["target"]
            rel_type = conn["type"]
            
            if source not in self.relationships:
                self.relationships[source] = {}
            
            if rel_type not in self.relationships[source]:
                self.relationships[source][rel_type] = []
            
            self.relationships[source][rel_type].append(target)
    
    def run_query(self, query, params=None):
        # Parse and execute query
        # This is a simplified implementation for evaluation
        results = []
        
        if "MATCH" in query and "CONNECTED_TO" in query:
            # Extract user ID if specified, or use default
            user_id = "user1"  # Default
            if "WHERE" in query and "userId" in query:
                # Extract userId from query (simplified)
                parts = query.split("userId")
                if len(parts) > 1:
                    id_part = parts[1].split("'")
                    if len(id_part) > 1:
                        user_id = id_part[1]
            
            # Get connections
            if user_id in self.relationships:
                for rel_type, targets in self.relationships[user_id].items():
                    if "CONNECTED_TO" in query or rel_type in query:
                        for target in targets:
                            if target in self.nodes:
                                results.append(self.nodes[target])
        
        return MockNeo4jResult(results)

# Updated evaluate_system function in evaluation.py
# Modify this section of your evaluation.py file
# Updated evaluate_system function with enhanced rate limit handling

def evaluate_system(dataset, queries=None, num_samples=5):
    """
    Evaluate the system on the given dataset and queries with improved error handling.
    
    Args:
        dataset: Dictionary containing users and connections
        queries: Dictionary of queries by category (simple, complex, ambiguous)
        num_samples: Number of queries to sample from each category
    
    Returns:
        Dictionary of evaluation results
    """
    # Setup components
    mongodb = EvalMockMongoDB(dataset)
    redis_client = setup_redis()
    neo4j_client = EvalMockNeo4j(dataset)
    
    model = setup_gemini()
    if not model:
        print("❌ Cannot continue without Gemini API key")
        return {}
    
    # Initialize LLM session with built-in retry logic
    llm_session = initialize_llm_session(model)
    
    db_clients = {
        "mongodb": mongodb,
        "neo4j": neo4j_client,
        "redis": redis_client
    }
    
    # Initialize metrics collectors
    metrics = {
        "query_translation": {
            "simple": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
            "complex": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0},
            "ambiguous": {"correct": 0, "partial": 0, "incorrect": 0, "total": 0}
        },
        "performance": {
            "simple": {"translation": [], "execution": [], "synthesis": []},
            "complex": {"translation": [], "execution": [], "synthesis": []},
            "ambiguous": {"translation": [], "execution": [], "synthesis": []}
        },
        "result_completeness": {
            "simple": [],
            "complex": [],
            "ambiguous": []
        },
        "response_quality": {
            "simple": [],
            "complex": [],
            "ambiguous": []
        },
        "errors": {
            "simple": [],
            "complex": [],
            "ambiguous": []
        },
        "raw_results": []
    }
    
    # Process queries
    if queries is None:
        queries = TEST_QUERIES
    
    # Enhanced adaptive rate limiting
    query_delay = 10.0  # Significantly increased initial delay between queries
    consecutive_errors = 0
    max_consecutive_errors = 3
    
    # Keep track of query progress across all categories
    total_queries = 0
    for category in queries:
        if num_samples > 0 and num_samples < len(queries[category]):
            total_queries += num_samples
        else:
            total_queries += len(queries[category])
    
    processed_queries = 0
    
    # Process each category
    for category_index, (category, query_list) in enumerate(queries.items()):
        # Sample queries if needed
        sampled_queries = query_list
        if num_samples > 0 and num_samples < len(query_list):
            sampled_queries = np.random.choice(query_list, num_samples, replace=False)
        
        # Add an initial delay between categories
        if category_index > 0:
            category_break = 60  # 1 minute break between categories
            print(f"Taking a {category_break} second break before processing {category} queries...")
            time.sleep(category_break)
        
        for query_index, query in enumerate(tqdm(sampled_queries, desc=f"Processing {category} queries")):
            # Add delay between queries to prevent rate limiting
            if query_index > 0:
                print(f"Waiting {query_delay:.1f}s before next query...")
                time.sleep(query_delay)
            
            # Setup components with retry logic
            translator = QueryTranslator(llm_session)
            executor = QueryExecutor(
                db_clients["mongodb"],
                db_clients["neo4j"],
                db_clients["redis"]
            )
            synthesizer = ResultSynthesizer(llm_session)
            
            try:
                # Step 1: Translate query
                start_time = time.time()
                query_plan = translator.translate_query(query)
                translation_time = time.time() - start_time
                metrics["performance"][category]["translation"].append(translation_time * 1000)  # Convert to ms
                
                # Reset consecutive errors counter on success
                consecutive_errors = 0
                
                # Gradually reduce delay if operations are succeeding
                query_delay = max(10.0, query_delay * 0.95)  # Minimum 10 second delay
                
                # Evaluate translation correctness
                translation_score = evaluate_translation(query, query_plan, category)
                if translation_score == "correct":
                    metrics["query_translation"][category]["correct"] += 1
                elif translation_score == "partial":
                    metrics["query_translation"][category]["partial"] += 1
                else:
                    metrics["query_translation"][category]["incorrect"] += 1
                
                metrics["query_translation"][category]["total"] += 1
                
                # Short pause between API operations
                time.sleep(2)
                
                # Step 2: Execute query
                start_time = time.time()
                results = executor.execute_query_plan(query_plan)
                execution_time = time.time() - start_time
                metrics["performance"][category]["execution"].append(execution_time * 1000)  # Convert to ms
                
                # Evaluate result completeness
                completeness = evaluate_completeness(query, results, category)
                metrics["result_completeness"][category].append(completeness)
                
                # Short pause between API operations
                time.sleep(2)
                
                # Step 3: Synthesize results
                start_time = time.time()
                final_response = synthesizer.synthesize_results(results, query)
                synthesis_time = time.time() - start_time
                metrics["performance"][category]["synthesis"].append(synthesis_time * 1000)  # Convert to ms
                
                # Evaluate response quality
                quality = evaluate_quality(query, final_response, results)
                metrics["response_quality"][category].append(quality)
                
                # Store raw results for detailed analysis
                metrics["raw_results"].append({
                    "category": category,
                    "query": query,
                    "query_plan": query_plan,
                    "results": results,
                    "response": final_response,
                    "metrics": {
                        "translation_score": translation_score,
                        "completeness": completeness,
                        "quality": quality,
                        "times": {
                            "translation": translation_time * 1000,
                            "execution": execution_time * 1000,
                            "synthesis": synthesis_time * 1000
                        }
                    }
                })
                
            except Exception as e:
                error_message = str(e)
                print(f"Error processing query '{query}': {error_message}")
                
                # Track error
                metrics["errors"][category].append({
                    "query": query,
                    "error": error_message
                })
                
                # Count as incorrect translation
                metrics["query_translation"][category]["incorrect"] += 1
                metrics["query_translation"][category]["total"] += 1
                
                # Adaptive backoff - increase delay on errors
                consecutive_errors += 1
                if consecutive_errors >= max_consecutive_errors:
                    # Take a much longer break after multiple consecutive errors
                    long_break = 180  # 3 minutes
                    print(f"⚠️ {consecutive_errors} consecutive errors detected. Taking a {long_break} second break...")
                    time.sleep(long_break)
                    consecutive_errors = 0
                    query_delay = min(60.0, query_delay * 2.0)  # Double delay up to 60 seconds
                else:
                    query_delay = min(30.0, query_delay * 1.5)  # Increase delay by 50% up to 30 seconds
            
            # Update processed queries counter
            processed_queries += 1
            
            # Calculate and display progress
            progress = (processed_queries / total_queries) * 100
            print(f"Overall progress: {processed_queries}/{total_queries} queries ({progress:.1f}% complete)")
    
    return metrics

# Helper functions for automated evaluation
def evaluate_translation(query, query_plan, category):
    """Evaluate the correctness of query translation"""
    try:
        # Check if query plan has expected structure
        if "query_type" not in query_plan or "databases" not in query_plan:
            return "incorrect"
        
        # Check if appropriate databases are selected
        if category == "simple" and len(query_plan["databases"]) > 1:
            # Simple queries should usually target one database
            return "partial"
        
        if category == "complex" and len(query_plan["databases"]) < 2:
            # Complex queries should target multiple databases
            return "partial"
        
        # Check if MongoDB query is present when MongoDB is in databases
        if "mongodb" in query_plan["databases"] and "mongodb_query" not in query_plan:
            return "partial"
        
        # Check if Neo4j query is present when Neo4j is in databases
        if "neo4j" in query_plan["databases"] and "neo4j_query" not in query_plan:
            return "partial"
        
        # Keyword matching for relevance
        query_lower = query.lower()
        relevant_keywords = []
        
        if "skills" in query_lower or "skill" in query_lower:
            relevant_keywords.append("skills")
        
        if "work" in query_lower or "experience" in query_lower or "job" in query_lower:
            relevant_keywords.append("experience")
            relevant_keywords.append("company")
        
        if "network" in query_lower or "connect" in query_lower:
            relevant_keywords.append("CONNECTED_TO")
        
        # Check if the query plan includes relevant fields
        mongodb_query = query_plan.get("mongodb_query", {})
        neo4j_query = query_plan.get("neo4j_query", "")
        
        matched_keywords = 0
        for keyword in relevant_keywords:
            if isinstance(mongodb_query, dict):
                # Check in MongoDB query
                if keyword in str(mongodb_query):
                    matched_keywords += 1
                    continue
            
            # Check in Neo4j query
            if keyword in neo4j_query:
                matched_keywords += 1
        
        # Calculate match ratio
        if not relevant_keywords:
            match_ratio = 1.0  # No keywords to match
        else:
            match_ratio = matched_keywords / len(relevant_keywords)
        
        if match_ratio >= 0.8:
            return "correct"
        elif match_ratio >= 0.5:
            return "partial"
        else:
            return "incorrect"
        
    except Exception as e:
        print(f"Error evaluating translation: {e}")
        return "incorrect"

def evaluate_completeness(query, results, category):
    """Evaluate completeness of query results"""
    # This would ideally be compared against ground truth
    # For automated evaluation, we'll use heuristics
    
    # Check if results contain data from expected databases
    mongodb_results = results.get("mongodb", [])
    neo4j_results = results.get("neo4j", [])
    
    if category == "simple":
        # For simple queries, check if we have at least some results
        if len(mongodb_results) > 0 or len(neo4j_results) > 0:
            return min(1.0, (len(mongodb_results) + len(neo4j_results)) / 5)  # Cap at 1.0
        return 0.0
    
    elif category == "complex":
        # For complex queries, we expect results from multiple databases
        if len(mongodb_results) > 0 and len(neo4j_results) > 0:
            return min(1.0, (len(mongodb_results) + len(neo4j_results)) / 10)  # Cap at 1.0
        elif len(mongodb_results) > 0 or len(neo4j_results) > 0:
            return 0.5  # Partial results
        return 0.0
    
    elif category == "ambiguous":
        # For ambiguous queries, any results are good
        if len(mongodb_results) > 0 or len(neo4j_results) > 0:
            return 0.7  # It's hard to determine completeness for ambiguous queries
        return 0.3
    
    return 0.0

def evaluate_quality(query, response, results):
    """Evaluate the quality of the synthesized response"""
    # This would ideally be done by human evaluators
    # For automated evaluation, we'll use simple heuristics
    
    # Check response length relative to results
    result_size = len(json.dumps(results))
    response_length = len(response)
    
    # Too short responses might not be informative
    if response_length < 20:
        return 1.0
    
    # Check if response mentions key entities from results
    mentions_count = 0
    
    # Extract entities from MongoDB results
    mongodb_results = results.get("mongodb", [])
    for result in mongodb_results:
        if "name" in result and result["name"] in response:
            mentions_count += 1
    
    # Extract entities from Neo4j results
    neo4j_results = results.get("neo4j", [])
    for result in neo4j_results:
        if "name" in result and result["name"] in response:
            mentions_count += 1
    
    # Calculate mention ratio
    total_entities = len(mongodb_results) + len(neo4j_results)
    mention_ratio = mentions_count / max(1, total_entities)
    
    # Calculate quality score (scale of 1-5)
    # Base score depends on response length relative to results
    base_score = min(5.0, max(1.0, (response_length / max(1, result_size)) * 10))
    
    # Adjust based on mention ratio
    mention_adjustment = mention_ratio * 2  # Scale to 0-2
    
    # Calculate final score
    quality_score = min(5.0, base_score + mention_adjustment)
    
    return quality_score

# Function to generate visualizations
def generate_visualizations(metrics):
    """Generate visualizations from evaluation metrics"""
    # Create a results directory
    os.makedirs("results", exist_ok=True)
    
    # 1. Query Translation Accuracy
    accuracy_data = []
    for category in ["simple", "complex", "ambiguous"]:
        cat_metrics = metrics["query_translation"][category]
        total = cat_metrics["total"]
        if total > 0:
            accuracy_data.append({
                "Category": category.capitalize(),
                "Fully Correct": cat_metrics["correct"] / total * 100,
                "Partially Correct": cat_metrics["partial"] / total * 100,
                "Incorrect": cat_metrics["incorrect"] / total * 100
            })
    
    accuracy_df = pd.DataFrame(accuracy_data)
    
    # Plot accuracy
    plt.figure(figsize=(12, 6))
    accuracy_plot = accuracy_df.plot(
        x="Category", 
        y=["Fully Correct", "Partially Correct", "Incorrect"],
        kind="bar", 
        stacked=True,
        color=["#4CAF50", "#FFC107", "#F44336"],
        title="Query Translation Accuracy by Query Type",
        figsize=(12, 6)
    )
    plt.ylabel("Percentage (%)")
    plt.ylim(0, 100)
    plt.xticks(rotation=0)
    plt.legend(loc="upper right")
    plt.tight_layout()
    plt.savefig("results/translation_accuracy.png", dpi=300)
    
    # Other visualization code...
    
    # Return data frames and summary
    return {
        "accuracy_df": accuracy_df,
        # Other return values
    }

# Main function to run the evaluation
def run_evaluation(sample_size=2, num_users=100, num_connections=500):
    """
    Run the complete evaluation pipeline.
    
    Args:
        sample_size: Number of queries to sample from each category
        num_users: Number of users in the synthetic dataset
        num_connections: Number of connections in the synthetic dataset
    
    Returns:
        Evaluation results
    """
    print(f"Generating synthetic dataset with {num_users} users and {num_connections} connections...")
    dataset = generate_synthetic_dataset(num_users=num_users, num_connections=num_connections)
    
    print(f"Evaluating system with {sample_size} queries from each category...")
    metrics = evaluate_system(dataset, num_samples=sample_size)
    
    print("Generating visualizations...")
    viz_results = generate_visualizations(metrics)
    
    print(f"Evaluation complete. Results saved to the 'results' directory.")
    return {
        "dataset": dataset,
        "metrics": metrics,
        "visualization_results": viz_results
    }