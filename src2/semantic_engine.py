from sentence_transformers import SentenceTransformer, util
import torch

class SemanticGlossMapper:
    def __init__(self):
        # Initialize a fast, efficient model for semantic similarity
        print("Loading Transformer model...")
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        
        # Define the master database of target Gloss structures
        # In a full system, this would be loaded from a JSON or SQL database
        self.gloss_database = {
            "BOOK MY": [
                "It is my book.",
                "I own the book.",
                "This book belongs to me.",
                "That is my book."
            ],
            "HOW YOU": [
                "How are you?",
                "How are you doing?",
                "How have you been?"
            ],
            "NAME YOU WHAT": [
                "What is your name?",
                "Who are you?",
                "May I have your name?"
            ]
        }
        
        self.intent_embeddings = self._precompute_embeddings()

    def _precompute_embeddings(self):
        """Encodes the master database into vector space for fast querying."""
        encoded_intents = {}
        for gloss, sentences in self.gloss_database.items():
            # Encode all variations for a specific gloss
            embeddings = self.model.encode(sentences, convert_to_tensor=True)
            encoded_intents[gloss] = embeddings
        return encoded_intents

    def translate_to_unified_gloss(self, input_text, threshold=0.65):
        """Matches input text to the closest semantic gloss structure."""
        input_embedding = self.model.encode(input_text, convert_to_tensor=True)
        
        best_match_gloss = None
        highest_score = 0.0

        for gloss, target_embeddings in self.intent_embeddings.items():
            # Calculate cosine similarity against all known variations
            cosine_scores = util.cos_sim(input_embedding, target_embeddings)[0]
            max_score = torch.max(cosine_scores).item()
            
            if max_score > highest_score:
                highest_score = max_score
                best_match_gloss = gloss

        # Return the match if it meets the confidence threshold
        if highest_score >= threshold:
            return best_match_gloss.split(), highest_score
        else:
            return ["UNKNOWN", "INTENT"], highest_score

# --- Execution Test ---
if __name__ == "__main__":
    mapper = SemanticGlossMapper()
    
    test_sentences = [
        "I own the book.",
        "It is my book.",
        "How are you doing today?",
        "The weather is nice." # An untrained concept
    ]
    
    print("\n--- Semantic Translation Results ---")
    for sentence in test_sentences:
        gloss, confidence = mapper.translate_to_unified_gloss(sentence)
        print(f"English: '{sentence}'")
        print(f"Target Gloss: {gloss} (Confidence: {confidence:.2f})\n")