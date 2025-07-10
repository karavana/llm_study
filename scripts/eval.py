# llm_case_study/scripts/eval.py

import os
import requests
import time
from typing import List

# --- CONFIGURATION ---
API_URL = "http://127.0.0.1:8000/query"
QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'questions.txt')
ANSWERS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'answers.txt')

def read_file_lines(filepath: str) -> List[str]:
    """Reads a file and returns a list of its lines, stripped of whitespace."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            # Filter out empty lines
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[✗] Error: File not found at {filepath}")
        return []

def query_rag_api(question: str) -> str:
    """Sends a question to the RAG API and returns the answer."""
    payload = {"question": question}
    try:
        response = requests.post(API_URL, json=payload, timeout=60) # 60-second timeout
        response.raise_for_status()  # Raise an exception for bad status codes (4xx or 5xx)
        return response.json().get("answer", "Error: 'answer' key not found in response.")
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed. {e}"

def main():
    """Main evaluation pipeline."""
    print("--- Starting RAG System Evaluation ---")

    # 1. Load questions and ground-truth answers
    print(f"[*] Loading questions from: {QUESTIONS_FILE}")
    questions = read_file_lines(QUESTIONS_FILE)
    
    print(f"[*] Loading answers from: {ANSWERS_FILE}")
    expected_answers = read_file_lines(ANSWERS_FILE)

    if not questions or not expected_answers:
        print("[✗] Cannot proceed without questions and answers. Exiting.")
        return

    if len(questions) != len(expected_answers):
        print("[✗] Error: The number of questions and answers do not match.")
        print(f"    Found {len(questions)} questions and {len(expected_answers)} answers.")
        return

    total_questions = len(questions)
    correct_predictions = 0
    total_time = 0

    # 2. Iterate through questions and evaluate
    for i, question in enumerate(questions):
        print("-" * 50)
        print(f"Evaluating Question {i+1}/{total_questions}:")
        print(f"  > Question: {question}")
        
        start_time = time.time()
        generated_answer = query_rag_api(question)
        end_time = time.time()
        
        elapsed_time = end_time - start_time
        total_time += elapsed_time

        expected_answer = expected_answers[i]
        
        print(f"  > Expected Answer: {expected_answer}")
        print(f"  > Generated Answer: {generated_answer}")
        print(f"  > Time Taken: {elapsed_time:.2f}s")

        # 3. Compare answers and calculate accuracy
        # A simple, case-insensitive comparison.
        # For a more robust evaluation, you might consider semantic similarity
        # or an LLM-as-a-judge approach.
        if generated_answer.strip().lower() == expected_answer.strip().lower():
            print("  > Result: [✓] Correct")
            correct_predictions += 1
        else:
            print("  > Result: [✗] Incorrect")

    print("=" * 50)
    print("--- Evaluation Complete ---")

    # 4. Report final accuracy
    if total_questions > 0:
        accuracy = (correct_predictions / total_questions) * 100
        avg_time = total_time / total_questions
        print(f"Final Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_questions})")
        print(f"Average Query Time: {avg_time:.2f}s")
    else:
        print("No questions were evaluated.")

if __name__ == "__main__":
    main()