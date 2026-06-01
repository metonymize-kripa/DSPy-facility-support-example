"""
Latency benchmark for all approaches.
Runs each model 100 times and reports p50, p95, p99 latencies.
"""

import json
import time
import sys
from pathlib import Path
import numpy as np

# Add parent to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# Test data - use a sample message
TEST_MESSAGE = "The printer on the 3rd floor is not responding. I need to print my payroll check by end of business today. Please fix this ASAP!"


def benchmark_rules(n=100):
    """Benchmark handwritten rules."""
    from _01_handwritten_rules.classify import classify_message
    
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        _ = classify_message(TEST_MESSAGE)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def benchmark_fasttext(n=100):
    """Benchmark FastText."""
    import fasttext
    
    model_dir = Path("data/models/fasttext")
    urgency_model = fasttext.load_model(str(model_dir / "urgency_model.bin"))
    sentiment_model = fasttext.load_model(str(model_dir / "sentiment_model.bin"))
    
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        _ = urgency_model.predict(TEST_MESSAGE)
        _ = sentiment_model.predict(TEST_MESSAGE)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def benchmark_svm(n=100):
    """Benchmark SVM."""
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    model_dir = Path("data/models/classical_ml")
    urgency_model = joblib.load(model_dir / "svm_urgency_model.pkl")
    sentiment_model = joblib.load(model_dir / "svm_sentiment_model.pkl")
    categories_model = joblib.load(model_dir / "svm_categories_model.pkl")
    vectorizer = joblib.load(model_dir / "svm_vectorizer.pkl")
    
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        X = vectorizer.transform([TEST_MESSAGE])
        _ = urgency_model.predict(X)
        _ = sentiment_model.predict(X)
        _ = categories_model.predict(X)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def benchmark_spacy_textcat(n=100):
    """Benchmark spaCy textcat."""
    import spacy
    
    model_path = Path("data/models/spacy_textcat")
    nlp = spacy.load(model_path)
    
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        doc = nlp(TEST_MESSAGE)
        _ = doc.cats
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def benchmark_entity_aug(n=100):
    """Benchmark entity-augmented spaCy."""
    import spacy
    import joblib
    from sklearn.feature_extraction.text import TfidfVectorizer
    
    nlp = spacy.load("en_core_web_sm")
    model_dir = Path("data/models/entity_augmented")
    clf = joblib.load(model_dir / "classifier.pkl")
    vectorizer = joblib.load(model_dir / "vectorizer.pkl")
    
    latencies = []
    for _ in range(n):
        start = time.perf_counter()
        doc = nlp(TEST_MESSAGE)
        # Extract entity features (simplified)
        features = {"has_equipment": 0, "has_location": 0, "has_urgency_word": 0}
        for ent in doc.ents:
            if ent.label_ in ["PRODUCT", "FAC"]:
                features["has_equipment"] = 1
        X = vectorizer.transform([TEST_MESSAGE])
        _ = clf.predict(X)
        latencies.append((time.perf_counter() - start) * 1000)
    return latencies


def print_stats(name, latencies):
    """Print latency statistics."""
    latencies = np.array(latencies)
    print(f"\n{name}")
    print("-" * 40)
    print(f"  p50: {np.percentile(latencies, 50):.2f} ms")
    print(f"  p95: {np.percentile(latencies, 95):.2f} ms")
    print(f"  p99: {np.percentile(latencies, 99):.2f} ms")
    print(f"  mean: {np.mean(latencies):.2f} ms")
    print(f"  min: {np.min(latencies):.2f} ms")
    print(f"  max: {np.max(latencies):.2f} ms")


def main():
    print("=" * 60)
    print("LATENCY BENCHMARK: Facility Support Classification")
    print("=" * 60)
    print(f"\nTest message: {TEST_MESSAGE[:60]}...")
    print(f"Iterations: 100 per approach\n")
    
    approaches = [
        ("01 Handwritten Rules", benchmark_rules),
        ("02 FastText", benchmark_fasttext),
        ("03 SVM (TF-IDF)", benchmark_svm),
        ("04 spaCy textcat", benchmark_spacy_textcat),
        ("04 Entity-Augmented spaCy", benchmark_entity_aug),
    ]
    
    results = {}
    for name, benchmark_fn in approaches:
        try:
            latencies = benchmark_fn(100)
            print_stats(name, latencies)
            results[name] = {
                "p50": float(np.percentile(latencies, 50)),
                "p95": float(np.percentile(latencies, 95)),
                "p99": float(np.percentile(latencies, 99)),
                "mean": float(np.mean(latencies)),
            }
        except Exception as e:
            print(f"\n{name}: FAILED ({e})")
    
    # Save results
    results_path = Path("data/results/07_latency_benchmark.json")
    results_path.write_text(json.dumps(results, indent=2))
    print(f"\n\nResults saved to: {results_path}")
    
    print("\n" + "=" * 60)
    print("SUMMARY TABLE")
    print("=" * 60)
    print(f"\n{'Approach':<30} {'p50 (ms)':<12} {'p95 (ms)':<12}")
    print("-" * 54)
    for name, stats in results.items():
        print(f"{name:<30} {stats['p50']:<12.2f} {stats['p95']:<12.2f}")


if __name__ == "__main__":
    main()
