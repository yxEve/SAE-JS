# Toward High Imperceptibility Deep JPEG Steganography Based on Sparse Adversarial Attack
## Overview 
This paper proposes an effective JPEG steganography framework. According to the vulnerability of deep learning models to adversarial examples. The method utilizes sparse adversarial examples to improve the security of stego images. Instead of merely confronting the steganalysis model designed for JPEG images, the spatial steganalysis model is also involved in adversarial training. To make the generated adversarial stego images more imperceptible, a visual perception loss is designed to fool human eyes.
## Train and Test Details
- Dataset: ALASKA2
- The encoder, decoder, perturbation generator network and discriminator in this paper are in "encoder.py", "decoder.py", "generator.py" and "critic.py"
- Use "main.py" to train the model
- Use "test.py" to test the model
- The "model.pt" in the directory sample_data is the saved model trained using random 01 matrics under 1bpp
## Note
Any questions can contact: 20211221062@nuist.edu.cn
