There are three python files for this project:
data.py:	Projects file 'input/train/TrainingData.txt' to first K Principal Components
net.py:		This, by default, trains an artificial neural network on the dataset,
		using K-Fold Cross-Validation, and presents average performance statistics, 
		then trains on the training dataset and predicts output on the testing dataset.
		The Architecture is defined by global variables at the top of the file, easily editable.

To project training dataset (input/train/TrainingData.txt) to K first Principal Components:
./data.py K
Output: Projected dataset (input/train/TrainingData_Project.txt), Singular values (output/Singular_Values.txt),
	Percent Variance of each singular value (output/Singular_Values_Percent_Variance.txt),

To run the ANN with K-Fold Cross-Validation on the Training Dataset:
./net.py [1]
	If the optional argument == 1, it will generate a completely new training and testing dataset.
Output: Validation statistics and a file of predicted categories saved in 'output/ANN_Predictions.txt'
