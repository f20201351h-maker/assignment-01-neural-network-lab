"""Regenerate every artifact the website uses, then run the adversarial checks.

    python experiments/run_all.py
"""
import time

import exp1_activations
import exp2_depth
import exp3_embeddings
import exp4_generalization
import validate

if __name__ == "__main__":
    t = time.time()
    for m in (exp1_activations, exp2_depth, exp3_embeddings, exp4_generalization):
        m.main()
    validate.main()
    print(f"done in {time.time() - t:.0f}s")
