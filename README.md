# Interpret.ai

This project uses a Convolutional Neural Network (CNN) to interpret handwritten medicine names from images. The project includes GUI for easy image upload and prediction.

## Files
- `gui.py`: Gradio GUI for image upload and prediction.
- `model.ipynb`: Jupyter notebook for training the CNN model.
- `README.md`: Project documentation.

## Usage
1. Train the model using `model.ipynb`.
2. Save the model as `model.h5`.
3. Save the label encoder classes as `classes.npy`.
4. Run `gui.py` to start the Gradio interface.

## Requirements
- TensorFlow
- Gradio
- OpenCV
- scikit-learn
- numpy
- matplotlib
- seaborn

## Installation
1. Clone the repository:
   ```sh
   git clone https://github.com/yourusername/medicine_interpret.ai.git
   cd interpret.ai

