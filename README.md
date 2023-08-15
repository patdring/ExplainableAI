# Explainable AI (XAI) and Computer Vision Techniques for Interpretable and Safe Autonomous Driving

This repository contains a collection of Jupyter notebooks that demonstrate various techniques for eXplainable AI (XAI) and Computer Vision in the context of autonomous driving. These notebooks were created using [Google Colaboratory](https://colab.research.google.com/), Python and Jupyter Notebooks.

## Notebooks

- `CaptionGeneration.ipynb`: This notebook demonstrates how to generate captions for images using a pre-trained transformer model (pre-trained transformers VisionEncoderDecoderModel).
- `DeepLift.ipynb`: This notebook shows how to use the DeepLift algorithm to interpret the predictions of a CNN (pre-trained pytorch vgg16).
- `FeatureAblation.ipynb`: This notebook demonstrates how to perform feature ablation to understand the importance of different features in a CNN (pre-trained pytorch vgg16).
- `FilterFeatureVisualization.ipynb`: This notebook shows how to visualize the filters and features learned by a CNN (pre-trained pytorch resnet50).
- `GradCam.ipynb`: This notebook demonstrates how to use Grad-CAM to generate class activation maps that highlight the important regions of an image for a given class (pre-trained pytorch resnet18).
- `GradientShap.ipynb`: This notebook demonstrates how to use GradientSHAP to explain the classification results of a instance segmentation model (pre-trained pytorch mask-rcnn_resnet50_fpn).
- `IntegratedGradients.ipynb`: This notebook demonstrates how to use Integrated Gradients to explain the classification results of a object detection model (pre-trained pytorch fasterrcnn_resnet50_fpn).
- `LIME.ipynb`: This notebook demonstrates how to use LIME (Local Interpretable Model-agnostic Explanations) to explain the predictions of a CNN (pre-trained pytorch resnet18).
- `ModelVisualizer.ipynb`: This notebook shows how to visualize the architecture of a deep learning model using Graphviz.
- `NoiseTunnel.ipynb`: This notebook shows how to use Noise Tunnel (with SmoothGrad) to explain the predictions of a CNN (pre-trained pytorch vgg11).
- `Occlusion.ipynb`: This notebook shows how to use occlusion to explain the predictions of a semantic segmentation model (pre-trained pytorch deeplabv3_resnet50).
- `TCAV.ipynb`: This notebook demonstrates how to use TCAV (Testing with Concept Activation Vectors) to interpret the predictions of a CNN (pre-trained pytorch resnet18).
- `TracInCPFast.ipynb`: This notebook shows how to use the TracInCPFast algorithm to interpret the predictions of a CNN (SimpleCNN).

## Usage

To use these notebooks, simply fetch and import them in Google Colaboratory and follow the instructions in each notebook. For more infos. please refer to https://research.google.com/colaboratory/faq.html. 
