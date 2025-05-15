# Adversarial Attacks with FGSM (Question 1)

## Overview
This project implements the Fast Gradient Sign Method (FGSM) and its Gaussian noise variant to generate adversarial examples for an MNIST classifier. It includes a FastAPI endpoint to operationalize the attack.

---

Files
File			Purpose
fgsm.py			Standard FGSM implementation
fgsm_gaussian.py	Gaussian FGSM variant
app_fgsm.py		FastAPI server and endpoint
mnist_model.py		MNIST CNN architecture
testing.py		To test and observe the output of Model
Output.doc		Screenshots of output

Usage
Generate Adversarial Examples
Via cURL:

curl -X POST -F "image=@test_image.png" -F "epsilon=0.1" "http://localhost:8000/attack?label=3"