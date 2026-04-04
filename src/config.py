import torch

CONFIG = {
    'input_folder': 'path/to/input/folder',
    'output_file': 'path/to/output.jsonl',
    'validated_output_file': 'path/to/validated_output.jsonl',
    'num_examples': 10000,
    'batch_size': 32,
    'max_workers': 4,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'models': {
        'gpt2': 'gpt2-medium',
        't5': 't5-base',
        'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
        'sentence': 'all-MiniLM-L6-v2'
    },
    # Text chunking parameters
    'chunk_size': 1500,       # target max characters per chunk
    'chunk_overlap': 200,     # overlap between consecutive chunks

    # Text processing parameters
    'keyword_count': 5,
    
    # Generation parameters
    'gpt2_max_length': 50,
    'no_repeat_ngram_size': 2,
    't5_max_length': 100,
    'tokenizer_max_length': 512,
    'top_k': 50,
    'top_p': 0.95,
    
    # Validation parameters
    'min_word_count': 10,
    'min_output_length': 50,
    'min_similarity_threshold': 0.3,
    'dedup_threshold': 0.95,  # cosine similarity above which outputs are considered duplicates
    'min_explanation_word_count': 30,
    'min_learning_path_steps': 3,
    'min_quiz_options': 4,
    'difficulty_keywords': ['easy', 'medium', 'difficult', 'challenging'],
    
    # Instruction types for dataset generation
    'instruction_types': [
        ("concept_explanation",
         "Explain the following concept in simple terms.\n\n"
         "Example:\n"
         "Concept: recursion\n"
         "Explanation: Recursion is when a function calls itself to solve a smaller version of the same problem. "
         "Imagine looking up a word in a dictionary and finding its definition uses another word you don't know — "
         "so you look that one up too. You keep going until you hit a word you already understand.\n\n"
         "Now explain:"),

        ("generate_question",
         "Generate a thought-provoking question about the following concept.\n\n"
         "Example:\n"
         "Concept: natural selection\n"
         "Question: If natural selection favors survival of the fittest, why do cooperative behaviors like altruism persist in many species?\n\n"
         "Now generate a question about:"),

        ("provide_example",
         "Provide a concrete, real-world example that illustrates the following concept.\n\n"
         "Example:\n"
         "Concept: opportunity cost\n"
         "Example: When a student spends Saturday studying for an exam instead of working a part-time shift, "
         "the opportunity cost is the wages they gave up — not just the time itself.\n\n"
         "Now provide an example for:"),

        ("learning_path",
         "Suggest a step-by-step learning path to master the following concept, listing prerequisite topics first and follow-up areas at the end.\n\n"
         "Example:\n"
         "Concept: machine learning\n"
         "Learning path:\n"
         "1. Prerequisites: Linear algebra, probability and statistics, Python programming\n"
         "2. Core topics: Supervised learning, gradient descent, model evaluation\n"
         "3. Intermediate: Neural networks, regularization, hyperparameter tuning\n"
         "4. Advanced follow-up: Deep learning, reinforcement learning, MLOps\n\n"
         "Now suggest a learning path for:"),

        ("misconception",
         "Identify a common misconception about the following concept and explain why it is wrong.\n\n"
         "Example:\n"
         "Concept: evolution\n"
         "Misconception: Many people think evolution means organisms consciously adapt to their environment.\n"
         "Correction: Evolution has no intent or direction. Traits that improve survival are passed on through reproduction over many generations — no individual organism changes its genes by trying.\n\n"
         "Now identify a misconception about:"),

        ("analogy",
         "Create a clear analogy that helps someone understand the following concept by comparing it to something familiar.\n\n"
         "Example:\n"
         "Concept: computer RAM\n"
         "Analogy: RAM is like your desk — it's the workspace where you keep everything you're actively using. "
         "Your hard drive is like a filing cabinet: much larger storage, but you have to walk over and retrieve things before you can work with them.\n\n"
         "Now create an analogy for:"),

        ("quiz_generation",
         "Generate a multiple-choice quiz question about the following concept. "
         "Include one correct answer and three plausible but incorrect distractors. Mark the correct answer.\n\n"
         "Example:\n"
         "Concept: photosynthesis\n"
         "Question: What is the primary source of energy that drives photosynthesis?\n"
         "A) Water\n"
         "B) Carbon dioxide\n"
         "C) Sunlight [CORRECT]\n"
         "D) Oxygen\n\n"
         "Now generate a quiz question about:"),

        ("concept_relation",
         "Explain how the following concept relates to another relevant concept in the same field.\n\n"
         "Example:\n"
         "Concept 1: supply\n"
         "Concept 2: demand\n"
         "Relation: Supply and demand are the two forces that together determine market price. "
         "When demand rises but supply stays the same, prices increase. When supply outpaces demand, prices fall. "
         "They act like two sides of a scale constantly seeking equilibrium.\n\n"
         "Now explain how this concept relates to another:"),

        ("application",
         "Describe a specific, practical application or real-world use case for the following concept.\n\n"
         "Example:\n"
         "Concept: Fourier transform\n"
         "Application: Audio engineers use Fourier transforms to break a sound recording into its individual frequency components. "
         "This allows them to isolate and boost specific frequencies (like bass) or remove unwanted noise from a recording.\n\n"
         "Now describe an application of:"),

        ("difficulty_assessment",
         "Assess the difficulty level of the following concept (easy / intermediate / advanced) and explain the specific reasons why learners find it challenging.\n\n"
         "Example:\n"
         "Concept: dynamic programming\n"
         "Difficulty: Advanced\n"
         "Reasons: Dynamic programming is hard for three reasons: (1) recognizing which problems have overlapping subproblems requires pattern-matching experience, "
         "(2) formulating the recurrence relation demands abstract thinking, and "
         "(3) translating the recurrence into correct bottom-up or top-down code requires careful implementation.\n\n"
         "Now assess the difficulty of:")
    ]
}
