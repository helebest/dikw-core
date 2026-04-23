"""Helper scripts for materialising public benchmarks into the
``evals/datasets/<name>/`` three-file shape (``dataset.yaml`` +
``corpus/`` + ``queries.yaml``) the runner already understands.

These are one-shot conversion scripts, not runtime plugins — once a
dataset directory exists, ``dikw eval --dataset <name>`` finds it
automatically and no extra code is needed.
"""
