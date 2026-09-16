"""Paper probe IDs, display names, and feature kinds in score-column order."""

METHOD_KINDS = {
    "mlp_baseline": "mlp",
    "kfold_pu": "mlp",
    "triplet_loss": "mlp",
    "attention_pooling": "attn",
    "attribute_conditioned_attention": "attn",
    "nnpu": "mlp",
    "dcpu": "mlp",
    "pu_ranking": "mlp",
}
METHOD_LABELS = {
    "mlp_baseline": "MLP",
    "kfold_pu": "K-Fold",
    "triplet_loss": "Triplet Loss",
    "attention_pooling": "Attention Pooling",
    "attribute_conditioned_attention": "Attribute-conditioned Attention",
    "nnpu": "nnPU",
    "dcpu": "DC-PU",
    "pu_ranking": "PURA",
}
METHOD_IDS = tuple(METHOD_KINDS)
