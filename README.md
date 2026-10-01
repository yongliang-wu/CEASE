# CEASE

Continual Concept Erasure in Diffusion Models by Suppressing Cross Edit Interference

Yongliang Wu, Haori Lu, Jinqi Luo, Wei Cao, Xingyu Zhu, Yaoyao Liu

[Project page](https://continual-erasure.cvmlgroup.web.illinois.edu/)

## Method

CEASE removes concepts sequentially from diffusion models without training. It preserves the shared replacement representation and projects new updates away from dominant directions of previous updates when interference is detected.

## Results

Experiments on continual erasure of celebrities, artistic styles, and instances across Stable Diffusion v1.4, v1.5, and v2.1 show a consistent balance between erasure and preservation. CEASE reduces accumulated interference and preserves unrelated generations across sequences of 100 edits.

## Code

Code coming soon.
