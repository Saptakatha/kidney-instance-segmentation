# Kidney Instance Segmentation

This repository provides a comprehensive solution for kidney instance segmentation using a weakly supervised learning approach. The workflow includes generating initial segmentation masks, training a model with both labeled and unlabeled data, and inferring on test images using the trained model.

### Installation <a name="install"></a>
First, clone the repository and navigate to the project directory:
```python
git clone https://github.com/Saptakatha/kidney-instance-segmentation.git
cd kidney-instance-segmentation
```

Install the required packages using ```requirements.txt```:
```python
pip install -r requirements.txt
```

### Generating Initial Kidney Instance Segmentation Masks <a name="generate_masks"></a>
To generate initial kidney instance segmentation masks, use the ```kidney_annotation.ipynb``` Jupyter Notebook. This notebook guides you through the process of loading images, setting seed points, and applying the region growing algorithm to create initial segmentation masks.

1. Open the Jupyter notebook:
```python 
jupyter notebook region_growing.ipynb
```
2. Follow the instructions in the notebook to:
    + Load the kidney images.
    + Set seed points for the region growing algorithm.
    + Generate and visualize the initial segmentation masks.

### Training the Model in a Weakly Supervised Manner <a name="train_model"></a>
Using the labeled image-mask pairs and unlabeled images, you can train a model in a weakly supervised manner with the ```weakly_supervised_train.py``` script.

1. Ensure your labeled and unlabeled data are organized in the appropriate directories.
2. Run the training script:
```python
python weakly_supervised_train.py --labeled_data_dir path/to/labeled_data --unlabeled_data_dir path/to/unlabeled_data --output_model_dir path/to/save_model
```
This script will:
+ Load the labeled and unlabeled data.
+ Generate pseudo-labels for the unlabeled data.
+ Train the model using a combination of supervised and unsupervised loss functions.

### Inferring on Test Images <a name="infer_model"></a>
Once the model is trained, you can use it to infer on test images with the ```weakly_supervised_infer```.py script.

1. Ensure your test images are organized in the appropriate directory.
2. Run the inference script:
```python
python weakly_supervised_infer.py --model_dir path/to/saved_model --test_data_dir path/to/test_data --output_dir path/to/save_predictions
```
The script will:
+ Load the trained model.
+ Perform inference on the test images.
+ Save the predicted segmentation masks to the specified output directory.

###  Conclusion <a name="conclusion"></a>
This repository provides a robust framework for kidney instance segmentation using weakly supervised learning. By following the steps outlined above, you can generate initial segmentation masks, train a model with both labeled and unlabeled data, and perform inference on new test images. For any questions or issues, please open an issue in the repository.