from collections import Counter
from typing import List, Dict, Any
from tqdm import tqdm
from utils import validation_failure


def validate_dataset(dataset: List[Dict[str, Any]], sentence_model) -> List[Dict[str, Any]]:
    """Keep examples that pass validation; print a breakdown of why the rest were dropped."""
    kept = []
    reasons = Counter()
    for example in tqdm(dataset, desc="Validating examples", unit="example"):
        reason = validation_failure(example["instruction_type"], example["output"], example["input"], sentence_model)
        if reason is None:
            kept.append(example)
        else:
            reasons[(example["instruction_type"], reason)] += 1

    print(f"Validation: {len(kept)} of {len(dataset)} examples passed")
    for (itype, reason), n in reasons.most_common():
        print(f"  dropped {n:>5}  {itype:<22} {reason}")
    return kept
