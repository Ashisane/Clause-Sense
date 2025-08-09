# test_hackrx.py
import time
import requests
from difflib import SequenceMatcher
from groq import Groq
import os
from dotenv import load_dotenv
import csv
from datetime import datetime

# Load env
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")
groq_client = Groq(api_key=GROQ_API_KEY)

# Local API endpoint
API_URL = "http://127.0.0.1:8000/hackrx/run"

# HackRx sample input
sample_input = {
    "documents": "https://hackrx.blob.core.windows.net/assets/policy.pdf?sv=2023-01-03&st=2025-07-04T09%3A11%3A24Z&se=2027-07-05T09%3A11%3A00Z&sr=b&sp=r&sig=N4a9OU0w0QXO6AOIBiu4bpl7AXvEZogeT%2FjUHNO7HzQ%3D",
    "questions": [
        "What is the grace period for premium payment under the National Parivar Mediclaim Plus Policy?",
        "What is the waiting period for pre-existing diseases (PED) to be covered?",
        "Does this policy cover maternity expenses, and what are the conditions?",
        "What is the waiting period for cataract surgery?",
        "Are the medical expenses for an organ donor covered under this policy?",
        "What is the No Claim Discount (NCD) offered in this policy?",
        "Is there a benefit for preventive health check-ups?",
        "How does the policy define a 'Hospital'?",
        "What is the extent of coverage for AYUSH treatments?",
        "Are there any sub-limits on room rent and ICU charges for Plan A?"
    ]
}

# Expected answers from HackRx docs
expected_answers = [
    "A grace period of thirty days is provided for premium payment after the due date to renew or continue the policy without losing continuity benefits.",
    "There is a waiting period of thirty-six (36) months of continuous coverage from the first policy inception for pre-existing diseases and their direct complications to be covered.",
    "Yes, the policy covers maternity expenses, including childbirth and lawful medical termination of pregnancy. To be eligible, the female insured person must have been continuously covered for at least 24 months. The benefit is limited to two deliveries or terminations during the policy period.",
    "The policy has a specific waiting period of two (2) years for cataract surgery.",
    "Yes, the policy indemnifies the medical expenses for the organ donor's hospitalization for the purpose of harvesting the organ, provided the organ is for an insured person and the donation complies with the Transplantation of Human Organs Act, 1994.",
    "A No Claim Discount of 5% on the base premium is offered on renewal for a one-year policy term if no claims were made in the preceding year. The maximum aggregate NCD is capped at 5% of the total base premium.",
    "Yes, the policy reimburses expenses for health check-ups at the end of every block of two continuous policy years, provided the policy has been renewed without a break. The amount is subject to the limits specified in the Table of Benefits.",
    "A hospital is defined as an institution with at least 10 inpatient beds (in towns with a population below ten lakhs) or 15 beds (in all other places), with qualified nursing staff and medical practitioners available 24/7, a fully equipped operation theatre, and which maintains daily records of patients.",
    "The policy covers medical expenses for inpatient treatment under Ayurveda, Yoga, Naturopathy, Unani, Siddha, and Homeopathy systems up to the Sum Insured limit, provided the treatment is taken in an AYUSH Hospital.",
    "Yes, for Plan A, the daily room rent is capped at 1% of the Sum Insured, and ICU charges are capped at 2% of the Sum Insured. These limits do not apply if the treatment is for a listed procedure in a Preferred Provider Network (PPN)."
]


# Accuracy helper
def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


# Groq fact-check
def fact_check(expected, predicted):
    prompt = f"""
You are a strict fact checker. 
Compare the following answers and classify:

Expected Answer: {expected}
Predicted Answer: {predicted}

Rules:
- Correct → meaning matches exactly, no wrong facts or missing key details
- Partially Correct → main point is right, but some details are missing or slightly inaccurate
- Incorrect → any key fact is wrong or contradicts the expected answer

Respond with only one word: Correct, Partially Correct, or Incorrect.
    """
    resp = groq_client.chat.completions.create(
        model="llama3-70b-8192",
        messages=[{"role": "user", "content": prompt}]
    )
    return resp.choices[0].message.content.strip()


# Save results to CSV
def save_results_to_csv(results, avg_similarity, correct_count, total_time):
    file_exists = os.path.isfile("test_results.csv")
    with open("test_results.csv", mode="a", newline="", encoding="utf-8") as file:
        writer = csv.writer(file)
        if not file_exists:
            writer.writerow(
                ["Timestamp", "Q#", "Question", "Expected", "Predicted", "Similarity", "Verdict", "Avg Similarity",
                 "Correct Count", "Total Time (s)"])

        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        for r in results:
            writer.writerow(
                [timestamp, r["q_num"], r["question"], r["expected"], r["predicted"], f"{r['similarity']:.2%}",
                 r["verdict"], f"{avg_similarity:.2%}", correct_count, f"{total_time:.2f}"])


if __name__ == "__main__":
    print("🚀 Sending request to local API for testing...")
    start_time = time.time()
    res = requests.post(API_URL, json=sample_input)
    end_time = time.time()

    if res.status_code != 200:
        print(f"❌ API request failed: {res.status_code} {res.text}")
        exit()

    answers = res.json()["answers"]
    total_similarity = 0
    correct_count = 0
    results = []

    print("\n===== Test Results =====")
    for i, (pred, exp) in enumerate(zip(answers, expected_answers)):
        score = similarity(pred, exp)
        total_similarity += score
        verdict = fact_check(exp, pred)
        if verdict == "Correct":
            correct_count += 1

        results.append({
            "q_num": i + 1,
            "question": sample_input["questions"][i],
            "expected": exp,
            "predicted": pred,
            "similarity": score,
            "verdict": verdict
        })

        print(f"\nQ{i + 1}: {sample_input['questions'][i]}")
        print(f"Expected: {exp}")
        print(f"Predicted: {pred}")
        print(f"Similarity: {score:.2%}")
        print(f"Groq Fact-Check Verdict: {verdict}")

    avg_similarity = total_similarity / len(expected_answers)
    total_time = end_time - start_time

    print("\n===== Summary =====")
    print(f"Average Similarity: {avg_similarity:.2%}")
    print(f"Correct Answers (Groq-verified): {correct_count}/{len(expected_answers)}")
    print(f"Total Response Time: {total_time:.2f} seconds")
    print(f"Avg Time per Question: {total_time / len(expected_answers):.2f} seconds")

    # Save to CSV
    save_results_to_csv(results, avg_similarity, correct_count, total_time)
    print("\n📊 Results saved to test_results.csv")
