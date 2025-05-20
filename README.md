# Pokemon Type Classification with Neural Networks
2 implementations of **Convolutional Neural Networks** to classify a Pokémon's type as *Water*, *Fire*, or *Grass*. The dataset contained many different Pokémon at different angles and colors (shiny variants included), these color images are 128x128 and *data augmentation* was used to better training. The labels were *One-hot encoded*. Each network was trained using a 70-15-15 train-validation-test split.

* Note: Each network was trained on Google Colab using [CUDA](https://docs.pytorch.org/docs/stable/cuda.html), then saved to a `.pt` file. These files were then loaded in to calculate metrics using [MPS (Metal Performance Shaders)](https://docs.pytorch.org/docs/stable/notes/mps.html) locally to avoid runtime timeouts and excessive usage on Colab, this difference in environment may affect results.

## Dataset
This dataset is from [Kaggle](https://www.kaggle.com/datasets/noodulz/pokemon-dataset-1000).

## KNN Baseline
**K-Nearest Neighbors** was used for a baseline multi-class classication.
- With using `k=112` neighbors, an average accuracy over 6 runs is `~49.8%`.

<img alt="KNN ROC" src="https://github.com/mangara22/PokemonTypeClassification/blob/main/roc.png">

## The Neural Networks
1) Custom CNN - `1,816,067` parameters
  <img alt="CNN Loss graph" src="https://github.com/mangara22/PokemonTypeClassification/blob/main/cnnloss.png">

2) Hybrid CNN with [ResNet18](https://docs.pytorch.org/vision/main/models/generated/torchvision.models.resnet18.html) - `263,171` parameters
  <img alt="Hyrbid CNN Loss graph" src="https://github.com/mangara22/PokemonTypeClassification/blob/main/hybridloss.png">

## Results
<img alt="CNN Confusion Matrix" src="https://github.com/mangara22/PokemonTypeClassification/blob/main/cm_cnn.png">

Accuracies over 20 runs (rounded to nearest tenth)
| Model | Training(%) | Testing(%) |
|-------|-------|-------|
| CNN   | 69.4  | 69.6 |
| Hybrid| 65.0  | 63.3  |
