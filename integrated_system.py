import re
import numpy as np
from typing import Dict, List, Any, Tuple
from collections import defaultdict
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import TruncatedSVD
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import warnings
from process_pdfs import process_pdf_files
import os
import json
from datetime import datetime
from summarizer import chat_offline
warnings.filterwarnings('ignore')


try:
    nltk.data.find('tokenizers/punkt')
    nltk.data.find('corpora/stopwords')
    nltk.data.find('corpora/wordnet')
except LookupError:
    print("Downloading required NLTK data...")
    nltk.download('punkt', quiet=True)
    nltk.download('stopwords', quiet=True)
    nltk.download('wordnet', quiet=True)

def find_relevant_sections(pdf_outline: Dict[str, Any], query_text: str, persona: str, 
                          max_sections: int = 10, min_relevance_score: float = 0.05) -> List[Dict[str, Any]]:
    """
    Find relevant sections from a PDF outline using NLP and ML techniques based on text similarity.
    
    Args:
        pdf_outline: Dictionary with 'title' and 'outline' keys
        query_text: Text to find similar content for
        persona: Persona string (e.g., 'Travel Planner', 'Software Tester')
        max_sections: Maximum number of sections to return
        min_relevance_score: Minimum relevance score threshold
    
    Returns:
        List of relevant sections with relevance scores, sorted by relevance
    """
    
    # Initialize NLP components
    lemmatizer = WordNetLemmatizer()
    stop_words = set(stopwords.words('english'))
    
    def preprocess_text(text: str) -> str:
        """Advanced text preprocessing using NLTK with fallback."""
        if not text or not text.strip():
            return ""
        
        try:
            # Convert to lowercase and tokenize
            tokens = word_tokenize(text.lower())
        except:
            # Fallback to simple split if NLTK tokenizer fails
            tokens = text.lower().split()
        
        # Remove non-alphabetic tokens and stopwords, then lemmatize
        processed_tokens = []
        for token in tokens:
            
            if (token.isalpha() and 
                len(token) >= 2 and 
                token not in stop_words):
                try:
                    lemmatized = lemmatizer.lemmatize(token)
                    processed_tokens.append(lemmatized)
                except:
                    # Fallback if lemmatization fails
                    processed_tokens.append(token)
        
        
        if not processed_tokens:
            
            cleaned = re.sub(r'[^a-zA-Z\s]', ' ', text.lower())
            cleaned_tokens = [word for word in cleaned.split() if len(word) >= 2 and word not in stop_words]
            return ' '.join(cleaned_tokens) if cleaned_tokens else text.lower()
        
        return ' '.join(processed_tokens)
    
    def create_context_query(persona: str, query_text: str, pdf_title: str = "") -> str:
        """Create a comprehensive context query for matching."""
        # Combine persona, query text, and PDF title for better context
        context_parts = [persona, query_text]
        if pdf_title:
            context_parts.append(pdf_title)
        
        return ' '.join(context_parts)
    
    
    sections = pdf_outline.get('outline', [])
    if not sections:
        return []
    
    section_texts = [section.get('text', '') for section in sections]
    
    
    processed_sections = [preprocess_text(text) for text in section_texts]
    pdf_title = pdf_outline.get('title', '')
    context_query = create_context_query(persona, query_text, pdf_title)
    processed_query = preprocess_text(context_query)
    
    # Filter out empty processed sections but be more lenient
    valid_indices = []
    valid_processed_texts = []
    
    for i, (original_text, processed_text) in enumerate(zip(section_texts, processed_sections)):
        # Keep section if either processed text has content OR original text is meaningful
        if (processed_text.strip() or 
            (original_text.strip() and len(original_text.strip()) > 1)):
            valid_indices.append(i)
            # Use processed text if available, otherwise use cleaned original
            if processed_text.strip():
                valid_processed_texts.append(processed_text)
            else:
                # Fallback cleaning for original text
                cleaned_original = re.sub(r'[^a-zA-Z\s]', ' ', original_text.lower())
                cleaned_original = ' '.join([w for w in cleaned_original.split() if len(w) >= 2])
                valid_processed_texts.append(cleaned_original or original_text.lower())
    
    if not valid_indices:
        # Last resort: include all sections with minimal processing
        print("Warning: No sections passed preprocessing. Using fallback approach.")
        valid_indices = list(range(len(sections)))
        valid_processed_texts = [text.lower() for text in section_texts]
    
    valid_sections = [sections[i] for i in valid_indices]
    
    # Create TF-IDF vectors with more lenient settings
    vectorizer = TfidfVectorizer(
        max_features=500,        
        ngram_range=(1, 2),      
        max_df=0.95,             
        min_df=1,                
        sublinear_tf=True,       
        token_pattern=r'\b[a-zA-Z]{2,}\b', 
        lowercase=True,
        strip_accents='ascii'
    )
    
    # Combine query and section texts for fitting the vectorizer
    all_texts = [processed_query] + valid_processed_texts
    
    try:
        # Fit vectorizer on all texts
        tfidf_matrix = vectorizer.fit_transform(all_texts)
        
        # Separate query vector from section vectors
        query_vector = tfidf_matrix[0:1]
        section_vectors = tfidf_matrix[1:]
        
        # Calculate cosine similarity between query and each section
        similarities = cosine_similarity(query_vector, section_vectors).flatten()
        
        
        if len(valid_sections) > 3 and tfidf_matrix.shape[1] > 10:
            try:
                n_components = min(50, min(tfidf_matrix.shape) - 1)
                if n_components > 0:  # Safety check
                    svd = TruncatedSVD(n_components=n_components, random_state=42)
                    lsa_matrix = svd.fit_transform(tfidf_matrix)
                    
                    query_lsa = lsa_matrix[0:1]
                    sections_lsa = lsa_matrix[1:]
                    lsa_similarities = cosine_similarity(query_lsa, sections_lsa).flatten()
                    
                    # Combine TF-IDF and LSA similarities (weighted average)
                    similarities = 0.7 * similarities + 0.3 * lsa_similarities
            except Exception as e:
                print(f"LSA processing failed, using TF-IDF only: {e}")
                # Continue with just TF-IDF similarities
                pass
    
    except Exception as e:
        print(f"Error in TF-IDF processing: {e}")
        # Fallback to simple keyword matching
        return fallback_keyword_matching(valid_sections, processed_query, max_sections, min_relevance_score)
    
    # Apply additional heuristic boosts
    boosted_similarities = apply_heuristic_boosts(
        similarities, valid_sections, persona, query_text, processed_query
    )
    
    # Ensure we always return some results if sections exist
    if len(boosted_similarities) > 0 and max(boosted_similarities) < min_relevance_score:
        # If no sections meet the threshold, lower it temporarily for this query
        adjusted_threshold = max(0.01, max(boosted_similarities) * 0.5)
        print(f"Adjusting relevance threshold from {min_relevance_score:.3f} to {adjusted_threshold:.3f}")
    else:
        adjusted_threshold = min_relevance_score
    
    # Create scored sections
    scored_sections = []
    for i, (section, similarity) in enumerate(zip(valid_sections, boosted_similarities)):
        if similarity >= adjusted_threshold:
            scored_sections.append({
                'section': section,
                'relevance_score': float(similarity),
                'level': section.get('level', ''),
                'text': section.get('text', ''),
                'page': section.get('page', 0)
            })
    
    # If still no results, return top 3 sections regardless of score
    if not scored_sections and valid_sections:
        print("No sections met relevance threshold. Returning top sections by similarity.")
        top_indices = np.argsort(boosted_similarities)[-3:][::-1]  # Top 3 in descending order
        for idx in top_indices:
            if idx < len(valid_sections):
                scored_sections.append({
                    'section': valid_sections[idx],
                    'relevance_score': float(boosted_similarities[idx]),
                    'level': valid_sections[idx].get('level', ''),
                    'text': valid_sections[idx].get('text', ''),
                    'page': valid_sections[idx].get('page', 0)
                })
    
    # Sort by relevance score (descending) and return top sections
    scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
    return scored_sections[:max_sections]

def apply_heuristic_boosts(similarities: np.ndarray, sections: List[Dict], 
                          persona: str, query_text: str, processed_query: str) -> np.ndarray:
    """Apply heuristic boosts to similarity scores based on section characteristics."""
    boosted_similarities = similarities.copy()
    
    # Safety check for empty similarities array
    if len(boosted_similarities) == 0:
        return boosted_similarities
    
    query_terms = set(processed_query.split())
    
    for i, section in enumerate(sections):
        if i >= len(boosted_similarities):  # Safety check
            break
            
        section_text_lower = section.get('text', '').lower()
        section_level = section.get('level', '')
        
        # Boost for structural importance (H1 > H2 > H3)
        if section_level == 'H1':
            boosted_similarities[i] *= 1.2
        elif section_level == 'H2':
            boosted_similarities[i] *= 1.1
        
        # Boost for key section types
        key_section_terms = [
            'introduction', 'overview', 'summary', 'objective', 'goal',
            'requirement', 'content', 'structure', 'methodology', 'approach'
        ]
        
        if any(term in section_text_lower for term in key_section_terms):
            boosted_similarities[i] *= 1.3
        
        # Boost for exact query term matches in section titles
        section_terms = set(section_text_lower.split())
        exact_matches = len(query_terms.intersection(section_terms))
        if exact_matches > 0:
            boosted_similarities[i] *= (1 + 0.2 * exact_matches)
        
        # Boost for numbered sections (often contain structured content)
        if re.search(r'^\d+\.', section_text_lower.strip()):
            boosted_similarities[i] *= 1.15
    
    return boosted_similarities

def fallback_keyword_matching(sections: List[Dict], processed_query: str, 
                             max_sections: int, min_relevance_score: float) -> List[Dict]:
    """Fallback method using simple keyword matching if TF-IDF fails."""
    query_terms = set(processed_query.split())
    scored_sections = []
    
    # If query is empty, use a different approach
    if not query_terms:
        query_terms = set(['introduction', 'overview', 'content'])  # Default terms
    
    for section in sections:
        section_text = section.get('text', '').lower()
        section_terms = set(section_text.split())
        
        # Calculate multiple similarity metrics with safety checks
        intersection = len(query_terms.intersection(section_terms))
        union = len(query_terms.union(section_terms))
        
        # Jaccard similarity with safety check
        jaccard_similarity = intersection / union if union > 0 else 0.0
        
        # Simple overlap similarity with safety check
        overlap_similarity = intersection / len(query_terms) if query_terms and len(query_terms) > 0 else 0.0
        
        # Combine similarities
        similarity = max(jaccard_similarity, overlap_similarity * 0.8)
        
        # Boost for important section types
        if any(term in section_text for term in ['introduction', 'overview', 'objective', 'content']):
            similarity *= 1.5
        
        scored_sections.append({
            'section': section,
            'relevance_score': similarity,
            'level': section.get('level', ''),
            'text': section.get('text', ''),
            'page': section.get('page', 0)
        })
    
    # Always return at least some results if sections exist
    if not any(s['relevance_score'] >= min_relevance_score for s in scored_sections) and scored_sections:
        # Return top 3 sections even if below threshold
        scored_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
        return scored_sections[:min(3, len(scored_sections))]
    
    # Filter by threshold and sort
    filtered_sections = [s for s in scored_sections if s['relevance_score'] >= min_relevance_score]
    filtered_sections.sort(key=lambda x: x['relevance_score'], reverse=True)
    return filtered_sections[:max_sections]




def aggregate_relevant_sections(output_dir, query_text, persona, input_documents, max_sections=5, max_subsections=5):
    """
    Aggregate relevant sections from multiple PDFs based on text similarity to a query.
    
    Args:
        output_dir: Directory containing processed PDF outlines
        query_text: Text to find similar content for
        persona: Persona context for the search
        input_documents: List of input document filenames
        max_sections: Maximum number of sections to return
        max_subsections: Maximum number of subsections to analyze
    
    Returns:
        Dictionary containing aggregated results with metadata, extracted sections, and summaries
    """
    all_sections = []
    all_subsections = []
    doc_to_sections = {}

    # For ranking importance
    section_candidates = []

    for filename in os.listdir(output_dir):
        if filename.endswith(".json"):
            file_path = os.path.join(output_dir, filename)
            with open(file_path, "r", encoding="utf-8") as f:
                pdf_outline = json.load(f)

            relevant_sections = find_relevant_sections(pdf_outline, query_text, persona, max_sections=10)
            docname = filename.replace(".json", ".pdf")
            doc_to_sections[docname] = relevant_sections

            for section in relevant_sections:
                section_candidates.append({
                    "document": docname,
                    "section_title": section["text"].strip(),
                    "page_number": section["page"],
                    "relevance_score": section["relevance_score"],
                    "section": section  # keep for possible subsection extraction
                })

    # Sort all section candidates by relevance and assign importance_rank
    section_candidates.sort(key=lambda x: x["relevance_score"], reverse=True)
    extracted_sections = []
    for i, sec in enumerate(section_candidates[:max_sections]):
        extracted_sections.append({
            "document": sec["document"],
            "section_title": sec["section_title"],
            "importance_rank": i + 1,
            "page_number": sec["page_number"]
        })

    # Subsection analysis: take top N sections and extract a "refined_text" (here, just use the text for demo)
    for sec in section_candidates[:max_subsections]:
        offline_summary = chat_offline(persona, query_text, sec['section']['section']['sub_content'])
        all_subsections.append({
            "document": sec["document"],
            "refined_text": offline_summary,  # Replace with actual refined text logic if needed
            "page_number": sec["page_number"]
        })

    result = {
        "metadata": {
            "input_documents": input_documents,
            "persona": persona,
            "query_text": query_text,
            "processing_timestamp": datetime.now().isoformat()
        },
        "extracted_sections": extracted_sections,
        "subsection_analysis": all_subsections
    }
    print(result)
    return result



if __name__ == "__main__":
    with open("challenge1b_input.json", "r", encoding="utf-8") as f:
        input_data = json.load(f)
        query_text = input_data["query_text"]["text"]  # Changed from job_to_be_done.task
        persona = input_data["persona"]["role"]
        input_documents = [doc["filename"] for doc in input_data["documents"]]

    output_dir = "./app/output"
    result = aggregate_relevant_sections(output_dir, query_text, persona, input_documents)
    
    with open("challenge1b_output.json", "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2, ensure_ascii=False)