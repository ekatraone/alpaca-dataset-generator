import torch

CONFIG = {
    'input_folder': 'data/input',
    'output_file': 'output/raw_dataset.jsonl',
    'validated_output_file': 'output/validated_dataset.jsonl',
    'num_examples': 1000,
    'batch_size': 8,
    'max_workers': 2,
    'device': torch.device("cuda" if torch.cuda.is_available() else "cpu"),
    'models': {
        'gpt2': 'gpt2-medium',
        't5': 't5-base',
        'sentiment': 'distilbert-base-uncased-finetuned-sst-2-english',
        'sentence': 'all-MiniLM-L6-v2'
    },
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
    'min_explanation_word_count': 30,
    'min_learning_path_stepspath_steps': 3,
    'min_quiz_options': 4,
    'difficulty_keywords': ['easy', 'medium', 'difficult', 'challenging'],
    
    # Instruction types for dataset generation
    'instruction_types': [
        ("concept_explanation", "Explain the following concept in simple terms:"),
        ("generate_question", "Generate a thought-provoking question about this concept:"),
        ("provide_example", "Provide a real-world example that illustrates this concept:"),
        ("learning_path", "Suggest a learning path to master this concept, including prerequisite topics and follow-up areas:"),
        ("misconception", "Identify and correct a common misconception about this concept:"),
        ("analogy", "Create an analogy to help understand this concept:"),
        ("quiz_generation", "Generate a multiple-choice quiz question about this concept, including the correct answer and three plausible distractors:"),
        ("concept_relation", "Explain how this concept relates to another relevant concept in the field:"),
        ("application", "Describe a practical application or use case for this concept:"),
        ("difficulty_assessment", "Assess the difficulty level of this concept and explain why it might be challenging for some students:")
    ]
}
