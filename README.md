# ML Mask Detection

This project uses machine learning to detect whether a person is wearing a mask or not.

The workflow includes:
1. **Data preprocessing**: Loading and cleaning images from the dataset.
2. **Model training**: Building a CNN using TensorFlow/Keras and training it on the dataset.
3. **Evaluation and testing**: Checking model accuracy on unseen images and predicting mask/no-mask.

## Dataset
You can download the dataset from this link:  
https://www.kaggle.com/datasets/omkargurav/face-mask-dataset


## How to Run

1. Clone the repository:
   ```bash
   git clone https://github.com/joysonnw/ml-mask-detection.git
   cd ml-mask-detection
2. Install the required dependencies:
   ```bash
   pip install -r requirements.txt
4. Train the model:
   ```bash
   python src/train.py
5. Test the model:
   ```bash
   python src/test.py

