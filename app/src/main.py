import sys

from crawler import crawl_urls, get_new_urls
from db import insert_passages_to_chromadb
from generate_response import evaluate_marketing_copy


def run_full_workflow(
    urls_path="urls.txt",
    output_path="output.jsonl",
    seen_path="seen_urls.txt",
    collection_name="pharma_copy_collection",
):
    print("Step 1: Checking for new FDA Regulatory Guidelines")
    new_urls = get_new_urls(urls_path, seen_path)

    if not new_urls:
        print("No new URLs found in urls.txt. Skipping crawl and DB insertion.")
    else:
        print(f"Found {len(new_urls)} new URL(s). Crawling and indexing FDA guidelines...")
        crawled_count = crawl_urls(new_urls, output_path, seen_path)
        if crawled_count > 0:
            print("Indexing FDA guidelines into ChromaDB")
            insert_passages_to_chromadb(output_path, collection_name)
        else:
            print("No new passages were generated during crawling. Skipping DB insertion.")

    print("\nStep 2: Pharmaceutical Marketing Copy Evaluation")
    print("Paste your pharmaceutical marketing copy below (press Enter twice when done):")
    lines = []
    while True:
        line = input()
        if line:
            lines.append(line)
        else:
            if lines:
                break
    
    marketing_copy = "\n".join(lines)
    
    if not marketing_copy.strip():
        print("No marketing copy provided.")
        return None
    
    print("\nEvaluating marketing copy against FDA guidelines...\n")
    evaluation = evaluate_marketing_copy(marketing_copy, collection_name)
    print("=== Compliance Evaluation Report ===")
    print(evaluation)
    return evaluation


if __name__ == "__main__":
    urls_path = sys.argv[1] if len(sys.argv) > 1 else "data/urls.txt"
    output_path = sys.argv[2] if len(sys.argv) > 2 else "data/output.jsonl"
    seen_path = sys.argv[3] if len(sys.argv) > 3 else "data/seen_urls.txt"
    collection_name = sys.argv[4] if len(sys.argv) > 4 else "pharma_copy_collection"

    run_full_workflow(urls_path, output_path, seen_path, collection_name)
