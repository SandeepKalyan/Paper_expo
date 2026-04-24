# Graph Report - Paper_expo  (2026-04-23)

## Corpus Check
- 23 files · ~1,029,581 words
- Verdict: corpus is large enough that graph structure adds value.

## Summary
- 98 nodes · 141 edges · 9 communities detected
- Extraction: 78% EXTRACTED · 22% INFERRED · 0% AMBIGUOUS · INFERRED: 31 edges (avg confidence: 0.75)
- Token cost: 0 input · 0 output

## Community Hubs (Navigation)
- [[_COMMUNITY_Community 0|Community 0]]
- [[_COMMUNITY_Community 1|Community 1]]
- [[_COMMUNITY_Community 2|Community 2]]
- [[_COMMUNITY_Community 3|Community 3]]
- [[_COMMUNITY_Community 4|Community 4]]
- [[_COMMUNITY_Community 5|Community 5]]
- [[_COMMUNITY_Community 6|Community 6]]
- [[_COMMUNITY_Community 7|Community 7]]
- [[_COMMUNITY_Community 8|Community 8]]

## God Nodes (most connected - your core abstractions)
1. `main()` - 11 edges
2. `RetinalDataset` - 9 edges
3. `main()` - 8 edges
4. `RSFConv2d` - 8 edges
5. `RSFConvBlock` - 7 edges
6. `RSFUNet` - 6 edges
7. `build_dataset()` - 5 edges
8. `ConvBlock` - 5 edges
9. `UNet` - 5 edges
10. `build_model()` - 5 edges

## Surprising Connections (you probably didn't know these)
- `main()` --calls--> `UNet`  [INFERRED]
  src/sanity_check.py → src/models/unet.py
- `evaluate()` --calls--> `compute_metrics_from_logits()`  [INFERRED]
  src/train.py → src/metrics/vessel_metrics.py
- `main()` --calls--> `load_config()`  [INFERRED]
  src/train.py → src/utils/config.py
- `main()` --calls--> `seed_everything()`  [INFERRED]
  src/train.py → src/utils/seed.py
- `main()` --calls--> `build_model()`  [INFERRED]
  src/train.py → src/models/factory.py

## Communities

### Community 0 - "Community 0"
Cohesion: 0.19
Nodes (10): DatasetPaths, _load_image(), _load_mask(), _read_ids(), RetinalDataset, Dataset, augment_train(), _clip_uint8() (+2 more)

### Community 1 - "Community 1"
Cohesion: 0.19
Nodes (7): ensure_dir(), write_json(), build_loss(), seed_everything(), evaluate(), main(), parse_args()

### Community 2 - "Community 2"
Cohesion: 0.23
Nodes (5): count_parameters(), RSFConvBlock, main(), RSFUNet, RSFUpBlock

### Community 3 - "Community 3"
Cohesion: 0.19
Nodes (6): ChaseDB1Dataset, DriveDataset, build_dataloader(), build_dataset(), RetinalDataset, StareDataset

### Community 4 - "Community 4"
Cohesion: 0.24
Nodes (4): build_model(), ConvBlock, UNet, UpBlock

### Community 5 - "Community 5"
Cohesion: 0.27
Nodes (8): load_config(), _apply_postprocess(), main(), parse_args(), _predict_probs(), compute_metrics_from_logits(), compute_metrics_from_probs(), _safe_div()

### Community 6 - "Community 6"
Cohesion: 0.39
Nodes (2): A compact RSF-style convolutional layer with Fourier-parameterized kernels., RSFConv2d

### Community 7 - "Community 7"
Cohesion: 0.52
Nodes (6): build_dataset(), draw_vessels(), main(), make_fov(), parse_args(), write_split()

### Community 8 - "Community 8"
Cohesion: 0.5
Nodes (1): Dataset loaders for retinal vessel segmentation.

## Knowledge Gaps
- **1 isolated node(s):** `A compact RSF-style convolutional layer with Fourier-parameterized kernels.`
  These have ≤1 connection - possible missing edges or undocumented components.
- **Thin community `Community 6`** (8 nodes): `A compact RSF-style convolutional layer with Fourier-parameterized kernels.`, `RSFConv2d`, `._equivariant_kernel()`, `.forward()`, `.__init__()`, `._spatial_kernel()`, `._transform_kernel()`, `.__init__()`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.
- **Thin community `Community 8`** (4 nodes): `__init__.py`, `__init__.py`, `Dataset loaders for retinal vessel segmentation.`, `__init__.py`
  Too small to be a meaningful cluster - may be noise or needs more connections extracted.

## Suggested Questions
_Questions this graph is uniquely positioned to answer:_

- **Why does `main()` connect `Community 1` to `Community 2`, `Community 3`, `Community 4`, `Community 5`?**
  _High betweenness centrality (0.376) - this node is a cross-community bridge._
- **Why does `build_dataloader()` connect `Community 3` to `Community 1`, `Community 5`?**
  _High betweenness centrality (0.351) - this node is a cross-community bridge._
- **Are the 8 inferred relationships involving `main()` (e.g. with `load_config()` and `seed_everything()`) actually correct?**
  _`main()` has 8 INFERRED edges - model-reasoned connections that need verification._
- **Are the 3 inferred relationships involving `RetinalDataset` (e.g. with `ChaseDB1Dataset` and `StareDataset`) actually correct?**
  _`RetinalDataset` has 3 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `main()` (e.g. with `load_config()` and `build_model()`) actually correct?**
  _`main()` has 4 INFERRED edges - model-reasoned connections that need verification._
- **Are the 4 inferred relationships involving `RSFConvBlock` (e.g. with `RSFUpBlock` and `RSFUNet`) actually correct?**
  _`RSFConvBlock` has 4 INFERRED edges - model-reasoned connections that need verification._
- **What connects `A compact RSF-style convolutional layer with Fourier-parameterized kernels.` to the rest of the system?**
  _1 weakly-connected nodes found - possible documentation gaps or missing edges._