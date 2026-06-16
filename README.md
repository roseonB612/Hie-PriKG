# Hie-PriKG: Hierarchical Privacy-Preserving Knowledge Graph Retrieval for Secure RAG Systems

This repository contains the reference implementation of **Hie-PriKG**, a hierarchical encrypted retrieval framework designed for privacy-preserving Knowledge Graph (KG) retrieval in Retrieval-Augmented Generation (RAG) systems.

Hie-PriKG combines **Function-Hiding Inner Product Encryption (FH-IPE)** and **Asymmetric Scalar Product Preserving Encryption (ASPE)** to achieve secure semantic retrieval while maintaining practical scalability and low-latency execution.

---

## Overview

Large-scale KG-enhanced RAG systems commonly outsource dense embeddings to cloud servers, creating potential privacy risks for both user queries and proprietary knowledge bases.

Hie-PriKG addresses this challenge through a two-stage retrieval architecture:

1. **FH-IPE-based cluster filtering**

   * Securely identifies candidate clusters.
   * Provides two-sided hiding for both query vectors and cluster centroids.

2. **ASPE-based entity ranking**

   * Performs lightweight similarity evaluation inside candidate clusters.
   * Preserves retrieval efficiency while reducing overall search complexity.

In addition, Hie-PriKG introduces an **Identity-Based Delegated Ciphertext Translation (IB-DCT)** mechanism that enables non-interactive multi-user retrieval without requiring continuous online participation from the KG owner.

---

## Repository Structure

```text
Hie-PriKG/
├── IPFE+ASPE/       # Python prototype implementation
├── FH-IPE-CPP/      # C++ implementation based on mcl and Eigen3
├── datasets/        # Processed benchmark datasets
├── figures/         # Experimental figures
└── README.md
```

### Python Prototype (`IPFE+ASPE`)

The Python implementation is intended for:

* Algorithm verification
* Security validation
* Mathematical correctness testing
* Rapid prototyping

Core components:

* Inner Product Functional Encryption (IPFE)
* ASPE-based secure ranking
* Hierarchical retrieval workflow

### C++ Implementation (`FH-IPE-CPP`)

The C++ implementation provides:

* FH-IPE construction based on bilinear pairings
* Efficient retrieval over large-scale datasets
* Multi-threaded execution
* Experimental evaluation used in the paper

Dependencies:

* mcl
* Eigen3
* OpenMP

---

## Building and Running

Please refer to the README file inside each subdirectory:

* `IPFE+ASPE/README.md`
* `FH-IPE-CPP/README.md`

---

## Citation

If you find this repository useful in your research, please cite:

```bibtex
@article{hieprikg2026,
  title={Hie-PriKG: Hierarchical Privacy-Preserving Knowledge Graph Retrieval for Secure RAG Systems},
  author={Yang, Keying and Wang, Tao and Ran, Jieru},
  year={2026},
  note={Under Review}
}
```
