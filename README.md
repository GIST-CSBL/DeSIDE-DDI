# DeSIDE-DDI

We developed a deep-learning strategy for interpretable prediction of DDIs that leverages drug-induced gene expression signatures (DeSIDE-DDI).
The model consists of two sub-models - feature generation model and DDI prediction model. Each model can be found in corresponding directory.

We provide example data, weights of constructed models for use, and toy example with Google Colab.

- Feature generation model
  - inputs: Compound fingerprints and properties
  - output: predicted gene expression
  - Example: 
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1m7fyZwFPp_85wKvFjbCwRCIOScrJEUtX?usp=sharing]


- DDI prediction model
  - inputs: predicted gene expressions of drug pairs
  - output: side effect score
  - Includes model construction and validation
  - DDI prediction Online with
  [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/drive/1XslE3XNsjm-dXwxrk_eVST6kALdwmvkd?usp=sharing]

  
- Feature_analysis (jupyter notebook)
  - How to extract significant genes given drug pairs
  - Visualization of changed latent features
  




Developed by Eunyoung Kim
