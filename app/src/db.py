import json
import chromadb
from chromadb.errors import NotFoundError

def load_passages_from_jsonl(jsonl_path):
    passages = []
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        for line in f:
            if line.strip():
                passages.append(json.loads(line))
    return passages

def insert_passages_to_chromadb(jsonl_path, collection_name="pharma_copy_collection"):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    try:
        collection = chroma_client.get_collection(name=collection_name)
    except NotFoundError:
        collection = chroma_client.create_collection(name=collection_name)

    passages = load_passages_from_jsonl(jsonl_path)

    documents = []
    ids = []
    metadatas = []
    seen_ids = set()
    duplicates_skipped = 0

    for p in passages:
        passage_index = p.get('passage_index', 0)
        pid = f"{p['url']}_{passage_index}"
        if pid in seen_ids:
            duplicates_skipped += 1
            continue
        seen_ids.add(pid)

        documents.append(p['text'])
        ids.append(pid)
        metadatas.append({
            'url': p['url'],
            'domain': p['domain'],
            'title': p['title'],
            'fetched_at': p['fetched_at'],
            'word_count': p['word_count']
        })

    if not documents:
        print(f"No unique passages to insert into collection '{collection_name}'")
        return

    existing_ids = set()
    if ids:
        try:
            existing_docs = collection.get(ids=ids)
            existing_ids = set(existing_docs.get("ids", []))
        except Exception:
            existing_ids = set()

    new_documents = []
    new_ids = []
    new_metadatas = []
    already_in_db = 0
    for document, pid, metadata in zip(documents, ids, metadatas):
        if pid in existing_ids:
            already_in_db += 1
            continue
        new_documents.append(document)
        new_ids.append(pid)
        new_metadatas.append(metadata)

    if not new_documents:
        print(f"No new passages to insert into collection '{collection_name}'. {already_in_db} passages were already stored.")
        return

    collection.upsert(documents=new_documents, ids=new_ids, metadatas=new_metadatas)
    print(
        f"Upserted {len(new_documents)} new passages into ChromaDB collection '{collection_name}'."
        f" Skipped {duplicates_skipped} duplicate IDs in JSONL and {already_in_db} already existing passages in the collection."
    )

def query_relevant_passages(user_question, n_results=3, collection_name="pharma_copy_collection"):
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name=collection_name)
    
    results = collection.query(query_texts=[user_question], n_results=n_results)
    return {
        "passages": results["documents"][0],
        "metadatas": results.get("metadatas", [])[0] if results.get("metadatas") else [],
        "ids": results.get("ids", [])[0] if results.get("ids") else [],
    }

def verify_pharma_copy(user_text, collection_name="pharma_copy_collection"):
    """
    Verify pharmaceutical copy against FDA standards using RAG.
    Retrieves relevant standards from the database and provides feedback.
    Returns a dict with 'relevant_standards': list of passages, 'feedback': str.
    """
    chroma_client = chromadb.PersistentClient(path="./chroma_db")
    collection = chroma_client.get_collection(name=collection_name)
    
    # Query for relevant FDA standards
    results = collection.query(query_texts=[user_text], n_results=5)
    relevant_passages = results["documents"][0]
    
    feedback = "Based on retrieved FDA standards, review your copy for compliance with the following requirements:\n"
    for i, passage in enumerate(relevant_passages, 1):
        feedback += f"{i}. {passage}\n"
    
    feedback += "\nCommon checks: Ensure fair balance of benefits and risks, truthful claims, no misleading statements, inclusion of prescribing information if applicable."
    
    return {
        "relevant_standards": relevant_passages,
        "feedback": feedback
    }

if __name__ == "__main__":
    # Example usage
    sample_copy = "Our new drug cures all headaches with no side effects! It's the best medicine ever."
    result = verify_pharma_copy(sample_copy)
    print("Relevant Standards:")
    for i, std in enumerate(result['relevant_standards'], 1):
        print(f"{i}. {std}")
    print("\nFeedback:")
    print(result['feedback'])