# MNIST Digit Classification with Convolutional Neural Networks
This project aims to compare the performance of hand-written digit classification Convolutional Neural Network (CNN) models in parallel and sequential modes. Two scripts, `classify_parallel.py` and `classify_sequential.py`, are provided for this purpose. The CNN model itself is implemented in the `CNN_MNIST.py` script.

## Getting Started

### Prerequisites
Make sure you have the following dependencies installed:
- Python (>=3.6)
- NumPy
- PyTorch
- TorchVision
- Matplotlib

### Installation
1. Clone this repository to your local machine using:
   ```bash
   git clone https://github.com/kcrmin/CNN_Hardware_Comparison.git
   ```
2. Install the required Python packages:
   ```bash
   pip install -r requirements.txt
   ```

### Usage
1. Run the program in **sequential mode**:
   ```bash
   python classify_sequential.py
   ```
2. Run the program in **parallel mode**:
   ```bash
   python classify_parallel.py
   ```

## Code Structure

### CNN_MNIST.py

- Defines a CNN model for MNIST digit classification.
- Provides functions for data loading and model initialization.

### classify_parallel.py

- Demonstrates parallel processing for predicting MNIST digits using the CNN model defined in `CNN_MNIST.py`.
- Utilizes the `multiprocessing` module to parallelize the prediction process, improving efficiency.

### classify_sequential.py

- Demonstrates sequential processing for predicting MNIST digits using the CNN model defined in `CNN_MNIST.py`.
- Does not use parallelization and processes predictions sequentially.

## File Structure

### mnist-cnn.pth

- Pre-trained model weights saved in PyTorch format.
- Although model was trained using CNN_MNIST, had to delete the training functions due to the readability.

### requirements.txt

- Contains the required Python packages to run the scripts.
- Enhance the usability as it simplifies the setup process.


### Configuration

Before running the scripts, you can configure the `meta_data` to adjust the parameters according to your requirements. Here's a brief overview of the `meta_data` configuration:

```python
   # Define meta_data
   meta_data["batch_size"] = 1
   meta_data["num_rows"] = 200
   meta_data["num_columns"] = 300
   meta_data["num_cells"] = meta_data["num_rows"] * meta_data["num_columns"]
   meta_data["threads"] = 2
```

## Results

After running each script, the following information will be printed:

- Number of threads used (only applicable for parallel mode)
- Total number of items classified
- Total runtime
- Accuracy

Additionally, the visualization of the mask image representing the classification results will be displayed.

## Screenshots

### Sequential
<img src = "https://github.com/kcrmin/CNN_Hardware_Comparison/assets/73128364/a82fcca0-d475-413a-95d9-d30ec46ea0a2">

### Parallel
<img src = "https://github.com/kcrmin/CNN_Hardware_Comparison/assets/73128364/5399aa43-13b6-4cf3-adb2-7920441b9bae">
