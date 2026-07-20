# Integrated Gradients (IG) GNN Attribution: Quick & Simple Guide

A quick reference guide explaining which Integrated Gradients (IG) wrapper code to use for PyTorch Geometric GNNs and why.

---

## Which Code Should You Use?

> **Use Snippet 2.** Snippet 1 has critical bugs that corrupt graph attributions.

---

## 3 Reasons Why Snippet 2 is Correct

### 1. Snippet 1 Breaks Graph Connections ⚠️
Captum tests 100 interpolation steps at once in a single large batch to save GPU time.
* **Snippet 1** passes 100 sets of node features but **only 1 set of edges**. This leaves 99% of your molecule's atoms completely disconnected during attribution calculation, causing wrong attributions.
* **Snippet 2** detects how many copies Captum created (`ratio = total_nodes // n_nodes`) and automatically duplicates the graph edges for all 100 copies.

### 2. Snippet 2 Attributes Both Atoms AND Fingerprints 🔬
* **Snippet 1** only calculates attributions for atom node features (`x`).
* **Snippet 2** calculates joint attributions for both atom features (`x`) and molecular fingerprints (`fp`), and plots a bar chart of the top active fingerprint bits.

### 3. Snippet 1 Has a Dataset Typo 🐛
* **Snippet 1** samples indices from `scaffold_results["test_dataset"]` but fetches data from `random_results["test_dataset"]`.
* **Snippet 2** consistently uses `results`.

---

## Quick Reference Code (Snippet 2)

```python
import io
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import torch
from captum.attr import IntegratedGradients
from rdkit import Chem
from rdkit.Chem import Draw
from rdkit.Chem.Draw import SimilarityMaps, rdMolDraw2D


# ---- Batching-aware wrapper ----
class GraphWrapper(torch.nn.Module):

    def __init__(self, model, edge_index, n_nodes):
        super().__init__()
        self.model = model
        self.edge_index = edge_index
        self.n_nodes = n_nodes

    def forward(self, x, fp):
        total_nodes = x.shape[0]
        ratio = total_nodes // self.n_nodes  # stacked copies by IG

        if ratio == 1:
            edge_index_batched = self.edge_index
            batch_idx = torch.zeros(
                self.n_nodes, dtype=torch.long, device=x.device
            )
        else:
            edge_index_batched = torch.cat(
                [self.edge_index + i * self.n_nodes for i in range(ratio)],
                dim=1,
            )
            batch_idx = torch.arange(
                ratio, device=x.device
            ).repeat_interleave(self.n_nodes)

        return self.model(
            atom_num=x, edges_index=edge_index_batched, fp=fp, batch=batch_idx
        )


def get_top_fp_bits(attr_fp, fp_vec, top_k=15):
    fp_scores = attr_fp.squeeze().detach().cpu().numpy()
    on_bits = np.nonzero(fp_vec)[0]
    if len(on_bits) == 0:
        return [], []
    on_scores = fp_scores[on_bits]
    order = np.argsort(-np.abs(on_scores))[:top_k]
    top_bits = on_bits[order]
    top_vals = on_scores[order]
    sort_idx = np.argsort(top_vals)
    return top_bits[sort_idx], top_vals[sort_idx]
```
