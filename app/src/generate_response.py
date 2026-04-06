from transformers import AutoTokenizer, AutoModelForCausalLM
import torch
from typing import Optional

from db import query_relevant_passages

# Model configuration
MODEL = "Qwen/Qwen2.5-7B-Instruct"  # Updated to a more recent model with better instruction-following capabilities

# Prompt template with mandatory source citations
PROMPT_TEMPLATE = """You are a pharmaceutical compliance assistant. Your job is to answer questions using ONLY the provided FDA regulatory context.

=== USER QUESTION ===
{user_question}

=== RETRIEVED FDA REGULATORY CONTEXT (from database) ===
{context}

=== CRITICAL INSTRUCTIONS ===
- ONLY use information explicitly stated in the context above.
- CITE YOUR SOURCES: Reference which passage(s) you used by including [Passage X] after each claim.
- Do NOT make up, infer, or add information not in the context.
- If the context does not directly address the question, say: "The provided context does not contain information to answer this question."
- Keep responses factual and grounded.

Answer with citations:"""

# Evaluation prompt for pharmaceutical marketing copy compliance
EVALUATION_PROMPT_TEMPLATE = """You are an FDA pharmaceutical compliance expert. Evaluate the provided marketing copy against the FDA regulatory guidelines provided.

=== MARKETING COPY TO EVALUATE ===
{marketing_copy}

=== APPLICABLE FDA REGULATORY GUIDELINES ===
{guidelines}

=== RULES FOR EVALUATION ===
- Treat the marketing copy as exact text. Do not infer approvals, indications, efficacy, or product claims that are not explicitly written.
- Do NOT read into the copy beyond the literal words it contains.
- If the copy contains only a call-to-action or brief directive with no explicit claims, state that clearly.
- Only use information from the copy and the provided guidelines.
- Cite guideline passages using [Passage X] for each non-compliance or risk.
- If no guideline passage supports your finding, do not report it as a violation.
- Do not create new claims or assume any FDA action unless it appears in the copy or the guidelines.
- Do not include any email signature, closing statement, document footer, or repeated "End of" text.

=== EVALUATION TASK ===
Analyze the marketing copy for FDA compliance. Provide:
1. COMPLIANT ASPECTS: What claims or statements follow FDA guidelines
2. NON-COMPLIANT ASPECTS: What violates FDA regulations (with citations)
3. RISKS/VIOLATIONS: Potential legal or regulatory violations
4. RECOMMENDATIONS: Specific suggestions to improve compliance

Compliance Evaluation Report:"""


class ResponseGenerator:
    """Manages lazy-loading and caching of the text generation model."""
    
    _instance: Optional["ResponseGenerator"] = None
    
    def __init__(self):
        self._tokenizer = None
        self._model = None
    
    @classmethod
    def get_instance(cls) -> "ResponseGenerator":
        """Get or create the singleton instance."""
        if cls._instance is None:
            cls._instance = cls()
        return cls._instance
    
    def _load_model(self) -> None:
        """Lazy-load the model and tokenizer on first use."""
        if self._model is not None:
            return  # Already loaded
        
        try:
            print(f"Loading model: {MODEL}")
            self._tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir="D:/HuggingFace/Cache")
            self._model = AutoModelForCausalLM.from_pretrained(MODEL, cache_dir="D:/HuggingFace/Cache")
            
            # Handle models without a pad token (e.g., gpt2)
            if self._tokenizer.pad_token is None:
                self._tokenizer.pad_token = self._tokenizer.eos_token
            
            # Move model to GPU if available, else CPU
            device = "cuda" if torch.cuda.is_available() else "cpu"
            self._model = self._model.to(device)
            self._model.eval()  # Set to evaluation mode
            
            print("Model loaded successfully")
        except Exception as e:
            raise RuntimeError(f"Failed to load model {MODEL}: {e}") from e
    
    def _generate_text(self, prompt: str) -> str:
        """Generate text using the model with explicit parameters."""
        inputs = self._tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048)
        device = "cuda" if torch.cuda.is_available() else "cpu"
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=500,
                do_sample=False,
                repetition_penalty=1.2,
                no_repeat_ngram_size=3,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )
        
        generated_text = self._tokenizer.decode(outputs[0], skip_special_tokens=True)
        return generated_text
    
    def generate(self, user_question: str, collection_name: str = "pharma_copy_collection") -> str:
        """Generate a response to a user question using retrieved context from the database with source citations."""
        self._load_model()
        
        results = query_relevant_passages(user_question, n_results=3, collection_name=collection_name)
        passages = results.get("passages", [])
        
        if not passages:
            return "I could not find any relevant information in the database."
        
        context = self._build_context(passages)
        prompt = PROMPT_TEMPLATE.format(user_question=user_question, context=context)
        
        try:
            generated_text = self._generate_text(prompt)
            # Remove the original prompt from the output if present
            answer = generated_text[len(prompt):].strip() if generated_text.startswith(prompt) else generated_text.strip()
            
            # Grounding validation: check for hallucinated claims
            hallucination_flags = self._detect_hallucination(answer, context, user_question)
            if hallucination_flags:
                print(f"[WARNING] Potential hallucination detected: {hallucination_flags}")
                answer = self._apply_safety_filter(answer, hallucination_flags)
            
            # Format cited passages for display
            cited_passages = self._extract_citations(answer, passages)
            formatted_response = self._format_with_citations(answer, cited_passages, passages)
            
            return formatted_response
        except Exception as e:
            raise RuntimeError(f"Failed to generate response: {e}") from e
    
    @staticmethod
    def _extract_citations(answer: str, passages: list) -> dict:
        """Extract which passages are cited in the answer."""
        import re
        cited = {}
        # Look for [Passage X] references in the answer
        pattern = r'\[Passage (\d+)\]'
        matches = re.findall(pattern, answer)
        for match in matches:
            idx = int(match) - 1  # Convert to 0-indexed
            if 0 <= idx < len(passages):
                cited[int(match)] = passages[idx]
        return cited
    
    @staticmethod
    def _format_with_citations(answer: str, cited_passages: dict, all_passages: list) -> str:
        """Format the answer with inline citations and source footnotes."""
        if cited_passages:
            footnotes = "\n\n=== SOURCES CITED ==="
            for passage_num in sorted(cited_passages.keys()):
                passage_text = cited_passages[passage_num]
                footnotes += f"\n\n[Passage {passage_num}]\n{passage_text[:200]}..."  # Show first 200 chars
            return answer + footnotes
        
        # Fallback: show the top retrieved passages when the model did not emit citations.
        footnotes = "\n\n=== RETRIEVED GUIDELINES (NO CITATIONS FOUND IN MODEL OUTPUT) ==="
        for i, passage_text in enumerate(all_passages, start=1):
            footnotes += f"\n\n[Passage {i}]\n{passage_text[:200]}..."
        footnotes += "\n\n[Note: The model did not include explicit citations in its response, so the top retrieved passages are shown above for reference.]"
        return answer + footnotes
    
    @staticmethod
    def _detect_hallucination(answer: str, context: str, user_question: str) -> list:
        """Detect potential hallucinations by checking for unsupported claims."""
        flags = []
        
        # Flag if answer makes strong definitive claims not in context
        strong_claims = ["FDA warns", "the FDA", "FDA rejects", "FDA approves", "is approved", "is rejected"]
        answer_lower = answer.lower()
        context_lower = context.lower()
        
        for claim in strong_claims:
            if claim in answer_lower and claim not in context_lower:
                flags.append(f"Unsupported claim: '{claim}'")
        
        return flags
    
    @staticmethod
    def _apply_safety_filter(answer: str, flags: list) -> str:
        """Apply safety filtering to reduce hallucination impact."""
        # If hallucination detected, ask user to verify with retrieved context
        if flags:
            return f"{answer}\n\n[Note: Please verify the above with the retrieved regulatory documents, as some claims may not be fully supported by the source context.]"
        return answer
    
    @staticmethod
    def _build_context(passages: list) -> str:
        """Format passages into numbered context with explicit passage labels."""
        return "\n".join([f"[Passage {i+1}] {passage}" for i, passage in enumerate(passages)])


def generate_response(user_question: str, collection_name: str = "pharma_copy_collection") -> str:
    """Public API for generating responses. Uses singleton instance internally."""
    generator = ResponseGenerator.get_instance()
    return generator.generate(user_question, collection_name)


def evaluate_marketing_copy(marketing_copy: str, collection_name: str = "pharma_copy_collection") -> str:
    """Evaluate pharmaceutical marketing copy against FDA guidelines from the database."""
    generator = ResponseGenerator.get_instance()
    generator._load_model()
    
    # Retrieve relevant FDA guidelines
    results = query_relevant_passages(marketing_copy, n_results=5, collection_name=collection_name)
    guidelines = results.get("passages", [])
    
    if not guidelines:
        return "No relevant FDA guidelines found in the database to evaluate this marketing copy."
    
    guidelines_text = generator._build_context(guidelines)
    evaluation_prompt = EVALUATION_PROMPT_TEMPLATE.format(
        marketing_copy=marketing_copy,
        guidelines=guidelines_text
    )
    
    try:
        generated_text = generator._generate_text(evaluation_prompt)
        # Remove the prompt from output
        evaluation = generated_text[len(evaluation_prompt):].strip() if generated_text.startswith(evaluation_prompt) else generated_text.strip()
        
        # Extract citations and format with footnotes
        cited_passages = generator._extract_citations(evaluation, guidelines)
        formatted_evaluation = generator._format_with_citations(evaluation, cited_passages, guidelines)
        
        return formatted_evaluation
    except Exception as e:
        raise RuntimeError(f"Failed to evaluate marketing copy: {e}") from e


if __name__ == "__main__":
    user_question = input("Enter consumer question: ")
    response = generate_response(user_question)
    print("\n=== Generated Response ===")
    print(response)
