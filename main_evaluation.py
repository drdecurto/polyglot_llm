# main_evaluation.py - Script to run experiments and generate results
# Dr. de Curtò ; BARCELONA Supercomputing Center / Universidad Pontificia Comillas / UOC
# Dr. de Zarzà; Universidad de Zaragoza / UOC

import os
import json
import time
import argparse
from evaluation import run_evaluation  # Now this function is properly defined at module level

def main():
    parser = argparse.ArgumentParser(description='Run experimental evaluation of polyglot persistence with LLMs')
    parser.add_argument('--api_key', type=str, help='Gemini API key')
    parser.add_argument('--sample_size', type=int, default=5, help='Number of queries to sample from each category')
    parser.add_argument('--num_users', type=int, default=100, help='Number of users in synthetic dataset')
    parser.add_argument('--num_connections', type=int, default=500, help='Number of connections in synthetic dataset')
    parser.add_argument('--output_dir', type=str, default='results', help='Directory to save results')
    
    args = parser.parse_args()
    
    # Set API key for Gemini if provided
    if args.api_key:
        os.environ['GEMINI_API_KEY'] = args.api_key
    else:
        print("Warning: No Gemini API key provided. Please set the GEMINI_API_KEY environment variable.")
    
    # Create output directory if it doesn't exist
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Record start time
    start_time = time.time()
    
    # Run the evaluation
    print(f"Starting evaluation with {args.sample_size} queries per category...")
    results = run_evaluation(
        sample_size=args.sample_size,
        num_users=args.num_users,
        num_connections=args.num_connections
    )
    
    # Record end time and calculate duration
    end_time = time.time()
    duration = end_time - start_time
    
    # Log experiment parameters
    experiment_log = {
        "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        "parameters": {
            "sample_size": args.sample_size,
            "num_users": args.num_users,
            "num_connections": args.num_connections
        },
        "duration_seconds": duration
    }
    
    # Save experiment log
    with open(os.path.join(args.output_dir, "experiment_log.json"), "w") as f:
        json.dump(experiment_log, f, indent=2)
    
    print(f"Evaluation completed in {duration:.2f} seconds")
    print(f"Results saved to {args.output_dir}")
    
    # Print summary results
    if "visualization_results" in results and "summary" in results["visualization_results"]:
        summary = results["visualization_results"]["summary"]
        print("\n===== EVALUATION SUMMARY =====")
        print(f"Overall Query Translation Accuracy: {summary['translation_accuracy']['overall']:.1f}%")
        
        print("\nAverage Processing Times (ms):")
        for category, times in summary["avg_processing_time"]["by_category"].items():
            print(f"  {category.capitalize()}: {times['total']:.0f} ms")
            print(f"    - Translation: {times['translation']:.0f} ms")
            print(f"    - Execution: {times['execution']:.0f} ms")
            print(f"    - Synthesis: {times['synthesis']:.0f} ms")
        
        print(f"\nOverall Result Completeness: {summary['result_quality']['overall_completeness'] * 100:.1f}%")
        print(f"Overall Response Quality: {summary['result_quality']['overall_quality']:.1f}/5.0")

if __name__ == "__main__":
    main()