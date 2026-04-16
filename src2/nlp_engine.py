import spacy

class TextToGlossProcessor:
    def __init__(self):
        # Load the pre-trained English pipeline
        self.nlp = spacy.load("en_core_web_sm")

    def process_basic_gloss(self, text):
        """
        A rule-based baseline to extract core concepts.
        This strips determiners, punctuation, and 'to be' verbs, 
        returning the capitalized root words (lemmas).
        """
        doc = self.nlp(text)
        gloss_sequence = []
        
        for token in doc:
            # Filter out punctuation, determiners (the, a), and copular verbs (is, am, are)
            if token.pos_ not in ['PUNCT', 'DET'] and token.lemma_ != 'be':
                # Sign language glosses are conventionally written in uppercase
                gloss_sequence.append(token.lemma_.upper())
                
        return gloss_sequence

# --- Execution Test ---
if __name__ == "__main__":
    processor = TextToGlossProcessor()
    
    test_sentences = [
    # --- Intent: HOW YOU ---
    "How are you?",
    "How are you doing?",
    "How have you been?",

    # --- Intent: BOOK MY ---
    "It is my book.",
    "I own the book.",
    "This book belongs to me.",
    "That is my book.",

    # --- Intent: NAME YOU WHAT ---
    "What is your name?",
    "Who are you?",
    "May I have your name?",

    # --- Expanded Intents (If implemented) ---
    "Hello",
    "Greetings",
    "Thank you",
    "I appreciate it",
    "Could you please",

    # --- Out-of-Bounds / Edge Cases (To test the 0.65 threshold) ---
    "The weather is nice.",
    "Moves",
    "I am going to the store."
    ]
    
    for sentence in test_sentences:
        gloss = processor.process_basic_gloss(sentence)
        print(f"English: '{sentence}' -> Baseline Gloss: {gloss}")