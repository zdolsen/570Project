How to Run
-There is a setup script called ProjectSetup that will walk through the necessary steps to run the project

Code that has been copied
-base folders such as models, utils, test_results, datasets

Code that has been modified/created
-Makefile
-model.py: I added a 5th layer to the network, so that it would more accuratly reflect the paper


New Files Created
-testNumWorkers.py: I created this script to run some dataloading tests ont he training data and display the optimal numberofworkers to train the data. This is important as the data takes very long to train for me.
-train.py: My training script. It trains the model from the WIDER Dataset. A lot of extra code was removed that didn't improve the model
-evaluateWithPictures.py: This takes in the model and overlays boxes over the faces then saves the photo. This has been buggy and has been very spotty to work.
-ProjectSetup.ipynb: Setup script for the project. Walks the user through setting up the code, installing the dependencies and datasets, and running.



The WIDER dataset was used and has been obtained from the link http://shuoyang1213.me/WIDERFACE/
It is a dataset containing data used to train and test faces


Video: https://youtu.be/4fmvX46m1m4