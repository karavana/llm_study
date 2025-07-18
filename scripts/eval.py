import os
import requests
import time
from typing import List
from google import genai

# --- CONFIGURATION ---
API_URL = "http://host.docker.internal:8000/query"
QUESTIONS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'questions.txt')
ANSWERS_FILE = os.path.join(os.path.dirname(__file__), '..', 'data', 'answers.txt')

# --- LLM-as-a-Judge CONFIGURATION (GOOGLE GENAI SDK) ---
# Load the Google API key from environment variables for security
GOOGLE_API_KEY = os.getenv("GOOGLE_API_KEY")
# We use a fast, capable, and cost-effective model as the judge.
JUDGE_MODEL = "gemini-1.5-flash-latest" # This remains the model identifier
judge_client = None

# --- Initialize Google GenAI Client ---
# We do this once at the start of the script for efficiency.
if GOOGLE_API_KEY:
    try:
        judge_client = genai.Client(api_key=GOOGLE_API_KEY)
        print(f"[✓] Successfully initialized Google GenAI client for judge model ({JUDGE_MODEL}).")
    except AttributeError:
        print("[!] Error: AttributeError encountered")
        print("    Evaluation will fall back to basic string matching.")
    except Exception as e:
        print(f"[!] Warning: Failed to initialize Google GenAI client: {e}")
        print("    Evaluation will fall back to basic string matching.")
else:
    print("[!] Warning: GOOGLE_API_KEY environment variable not found.")
    print("    Evaluation will fall back to basic string matching.")


def read_file_lines(filepath: str) -> List[str]:
    """Reads a file and returns a list of its lines, stripped of whitespace."""
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            return [line.strip() for line in f if line.strip()]
    except FileNotFoundError:
        print(f"[✗] Error: File not found at {filepath}")
        return []

def query_rag_api(question: str) -> str:
    """Sends a question to the RAG API and returns the answer."""
    payload = {"question": question}
    try:
        response = requests.post(API_URL, json=payload, timeout=60)
        response.raise_for_status()
        return response.json().get("answer", "Error: 'answer' key not found in response.")
    except requests.exceptions.RequestException as e:
        return f"Error: API request failed. {e}"


def is_correct_llm_judge(question: str, generated_answer: str, expected_answer: str) -> bool:
    """
    Uses Google Gemini to judge if the generated answer is correct, using the new genai.Client.
    """
    if not judge_client:
        print("    -> Judge: Basic string matching (fallback)")
        return generated_answer.strip().lower() == expected_answer.strip().lower()

    prompt = f"""
You are an impartial AI evaluator. Your primary goal is to validate if the core factual information in the "Generated Answer" matches the "Expected Answer".

**EVALUATION RULES:**
- The "Generated Answer" does NOT need to be a complete sentence or grammatically perfect.
- It does NOT need to be a word-for-word match.
- Be VERY FLEXIBLE. If it's semantically correct, then it is correct.
- Focus on the factual accuracy based on the provided "Expected Answer".

---
**EXAMPLE:**
- Original Question: What is the capital of France?
- Expected Answer: The capital of France is Paris, a major European city.
- Generated Answer: Paris
- Your Decision: CORRECT
---

**TASK TO EVALUATE:**

**Original Question:**
{question}

**Expected Answer:**
{expected_answer}

**Generated Answer:**
{generated_answer}

---
Based on the rules and example above, is the "Generated Answer" factually correct?
Your response MUST be a single word: either "CORRECT" or "INCORRECT".
"""
    try:
        response = judge_client.models.generate_content(
            model=f"{JUDGE_MODEL}",
            contents=prompt
    )
        decision = response.text.strip().upper()
        
        print(f"    -> Judge: Gemini says '{decision}'")

        if "CORRECT" == decision:
            return True
        elif "INCORRECT" == decision:
            return False
        else:
            print("    -> Judge: [!] Warning: Unexpected response from judge. Defaulting to INcorrect.")
            return False

    except Exception as e:
        print(f"    -> Judge: [✗] Error during Gemini API call: {e}")
        print("    -> Judge: Falling back to basic string matching for this question.")
        return generated_answer.strip().lower() == expected_answer.strip().lower()


def main():
    """Main evaluation pipeline."""
    print("--- Starting RAG System Evaluation ---")

    print(f"[*] Loading questions from: {QUESTIONS_FILE}")
    questions = read_file_lines(QUESTIONS_FILE)
    
    print(f"[*] Loading answers from: {ANSWERS_FILE}")
    expected_answers = read_file_lines(ANSWERS_FILE)

    if not questions or not expected_answers:
        print("[✗] Cannot proceed without questions and answers. Exiting.")
        return

    if len(questions) != len(expected_answers):
        print(f"[✗] Error: The number of questions and answers do not match.")
        print(f"    Found {len(questions)} questions and {len(expected_answers)} answers.")
        return

    total_questions = len(questions)
    correct_predictions = 0
    total_time = 0

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
        
        print(f"  > Expected Answer:  {expected_answer}")
        print(f"  > Generated Answer: {generated_answer}")
        print(f"  > Time Taken: {elapsed_time:.2f}s")
        result = is_correct_llm_judge(question, generated_answer, expected_answer)
        print(result)
        if result:
            print("  > Result: [✓] Correct")
            correct_predictions += 1
        else:
            print("  > Result: [✗] Incorrect")
            
    print("=" * 50)
    print("--- Evaluation Complete ---")

    if total_questions > 0:
        accuracy = (correct_predictions / total_questions) * 100
        avg_time = total_time / total_questions
        print(f"Final Accuracy: {accuracy:.2f}% ({correct_predictions}/{total_questions})")
        print(f"Average Query Time: {avg_time:.2f}s")
    else:
        print("No questions were evaluated.")

if __name__ == "__main__":
    main()