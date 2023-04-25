# Authors
Cierra Koen
_Machine Learning_
_Georgia State University_
Atlanta, Georgia
 ckoen1@student.gsu.edu
 
 Ikenna Okonkwo
_Machine Learning_
_Georgia State University_
Atlanta, Georgia
 iokonkwo2@student.gsu.edu

Omari mari Ward
_Machine Learning_
_Georgia State University_
Atlanta, Georgia
 oward3@student.gsu.edu
 
# Hotel Cancellations



**_Abstract_— Bookings for hotels, which are frequently made in advance by tourists looking for a cozy and dependable place to stay, are extremely important to the hospitality business. Nevertheless, even if a room is reserved, a cancellation may occur due to a variety of reasons. We used various machine learning techniques and validation splits to examine Jesse Mostipak's "Hotel Booking Demand" Kaggle dataset to pinpoint the causes influencing this trend. Identifying a binary classification model with a high accuracy and f1-score or a R2 score and mean standard error that can predict possible hotel cancellations was our main objective. By doing so, we hoped to get knowledge that may aid hotel managers in streamlining their operations and raising client happiness.**

_ **Keywords—Hotel Cancellation, Hotel Booking Demand, Binary Classification, Machine Learning** _

# I.Introduction

Our project idea is to predict whether a hotel booking will be cancelled based on various features: arrival date, number of adults and children, lead time, the country, previous bookings not cancelled, reserved room type, assigned room type, booking changes, deposit type, the agent, company, days on the waiting list, customer type, average daily rate, etc. For this prediction, we are using the Kaggle set "Hotel booking demand" developed by Jesse Mostipak [2]. We will use 13 different algorithms and make sure each algorithm has a train and test split of either 80-20, 70-30, or 50-50 and evaluate which split is the best. In the end, we want to find a model that does not overfit and still accurately predicts whether someone's hotel reservation would be cancelled or not. Each model will give us a binary classification of 0 being not cancelled while 1 is seen as a cancellation.

# II.Survey of Related work

Below we are providing different academic articles and explanations on why they are relevant to our project.

## A."Predicting hotel booking cancellations to decrease uncertainty and increase revenue"

This paper seeks to address the issues caused by booking cancellations. To do so, the authors utilized datasets from four resort hotels in Algarve, Portugal, and constructed multiple models using five different classification algorithms. The dataset consisted of 38 features and a total of 33,141 observations from all four hotels. The classification algorithms which were chosen include Boosted Decision Tree, Decision Forest, Decision Jungle, Locally Deep Support Vector Machine, and Neural Networks. The features had varying weights and contributions for each hotel, thus necessitating the construction of individual models for each hotel. K-fold cross-validation was employed to measure each algorithm's performance on all four hotels. The results on the test sets indicated that Decision Forest algorithm was the best performing classifier with an accuracy of 91%, 98%, 97%, and 92% on all four hotels respectively. This article effectively analyzes a complex data set and builds a model with remarkable accuracy on both the training and test sets. Consequently, this paper demonstrates the importance of ensemble algorithms such as Boosted Trees, and Decision Forests in overcoming the issue of overfitting and constructing a model good enough to generalize on novel test data [4].

## B."Prediction of Hotel Booking Cancellation using Deep Neural Network and Logistic Regression Algorithm"

This article examines a dataset with 13 independent variables and analyzes the correlation between hotel reservation cancellations and the growth of international tourism. The study was conducted by N. A. Putro et al. and involved the implementation of four distinct algorithms, including DNN, Logistic Regression, Three-way split method AUC-ROC, and Encoder-Decoder. Notably, the Three-way split method achieved an AUC-ROC score that was 15.03% higher than the Decision Tree, 2.93% higher than the KNN, and 2.38% higher than the Logistic Regression, while Logistic Regression proved to be the most effective algorithm in terms of optimization. Overall, this paper is a great resource for our project since the article covers many Machine Learning algorithms that we haven't proposed. We can even use Logistic Regression as our test hypothesis when comparing it to this paper [3].

## C."End-to-End Hotel Booking Cancellation Machine Learning Model"

This paper uses machine learning algorithms to make a hotel booking cancellation prediction model. This article and data were conducted by Suvarana Gawali. The data set includes 119,390 observations from both a city and a resort hotel in Portugal from the years 2015 to 2017. The algorithms mentioned in this dataset are K-Nearest Neighbor, Logistic Regression, Decision Tree, and Random Forest. It was determined in the article that random forest and decision tree were the best algorithms to use in the data regarding accuracy. K Nearest had an accuracy of 89%, Decision Tree had 95%, Random Forest had 95%, and Logistic Regression had 81% accuracy. This article would be very useful for our paper because it is initially being used to show how machine learning plays such a crucial role in the applications that we use today [1].

# III.Materials and Methods

## A.Data explanation and characterization

The data in question is derived from an article entitled "Hotel Booking Demand Datasets", which was composed by Nuno Antonio, Ana Almeida, and Luis Nunes. The dataset utilized in this particular study was procured from Kaggle and had undergone some form of preliminary cleansing. It comprises authentic booking information from two hotels spanning the years 2015 to 2017. The dataset is comprised of various attributes, including but not limited to the hotel name, year of arrival, month of arrival, number of adults, infants, children, number of previous cancellations, and so forth. In total, the dataset contains 119,390 samples and 32 features. According to the data, the proportion of non-cancelled bookings to cancelled bookings is 63 to 37. This indicates that the dataset exhibits skewed class proportions and is mildly imbalanced. It is important to note that this mild imbalance may have slight implications for any subsequent analysis or modeling efforts. For the purpose of classification, only 20% of the data was utilized in order to significantly reduce runtime. The imbalance in the reduced subsample was maintained for consistency.

## B.Data preprocessing

There exist 20 numerical features and 12 categorical features within the dataset. In order to ensure the quality of the dataset, a thorough scan was conducted to detect duplicates, outliers, strange values, and missing values. No duplicates were found within the dataset. However, a small percentage of outliers were detected within the numerical attributes. Despite this, we have decided against clipping or dropping the samples with outliers as they may be of utmost importance in predicting the cancellations of bookings. Conversely, there were missing values in the dataset in four attributes. Specifically, the children's attribute contained 4 missing values. In order to rectify this, the missing values were replaced with a zero as N/A values most likely indicate that the booking has no children as guests. Furthermore, the country attribute contained 488 missing values. We have decided against imputing this attribute as it could cause our model to believe that samples with the missing values had the same country of origin. The column in question exhibited a minimal percentage of missing values. Consequently, samples with the missing country values were dropped. The agent attribute contained 16430 missing values, while the company attribute contained 112593 null values. These missing values were replaced with a value of 0, indicating that the booking was made without an agent or company, respectively. In the dataset, there were peculiar samples where the sum of adults, babies, and children attributes was zero. This anomaly most likely indicated an error either in the booking or data collection. As these samples were deemed meaningless, they were also dropped from the dataset.

Correlation analysis was performed on the features using the Pearson coefficient. The strongest features according to the correlation matrix were lead time (number of days between date booking was created and arrival date), number of special requests, required car parking spaces, and booking changes. Upon further analysis of the dataset, it was discovered that a particular feature had the potential to create data leakage in our machine learning models. Specifically, the feature in question was the reservation status, which was found to be highly correlated with the label. In order to prevent this leakage from occurring during model training, both the reservation status and the reservation status date were dropped from the dataset, as they contained information about the target variable. After data cleaning, the resulting dataset size was down to 118,732 samples and 30 columns. Data transformation was the final step in the preprocessing stage. Numerical attributes underwent z-score normalization, which involved standardizing all values in each attribute's mean and standard deviation. Meanwhile, categorical features were encoded through a one-hot encoding scheme that generated a binary column for each category. Together, these preprocessing techniques optimized the data for use in subsequent analysis.

# IV.Key Algorithms

As previously mentioned, in order to reduce runtime, only a fraction of the cleaned data was utilized for model training and evaluations. Specifically, a mere 20% of the data was used, resulting in a reduced sample size of 23,746 samples and 30 features. This smaller dataset was then divided into two distinct parts: the inputs and the label. The label, which is a binary feature named is\_cancelled, serves as the basis for classification and consists of 23,746 samples and 1 feature. The inputs, on the other hand, encompass all attributes except for the label and comprise a total of 23,746 samples and 29 features. In total, thirteen algorithms were employed in predicting cancellations, including eleven classification algorithms and two regression algorithms. To facilitate the training and testing of these models, the dataset was divided into two separate sets: a training set and a test set. All of the algorithms were subjected to training and testing on three distinct sampling evaluations. The first sample consisted of an 80% split for training data and a 20% split for the test set. The second sample was a 70-30 split for the training and test sets, respectively. The final sample was split 50-50 for training and testing sets. In total, 39 models were trained, with three per algorithm on the three respective sample splits. For each model, learning curves were obtained to assess the algorithm's ability to generalize properly on the provided dataset. Prior to training, all models underwent a pipeline process where the transformation procedure (z-score normalization & one-hot encoding) explained previously was carried out on the inputs. The thirteen algorithms used are listed below.

## A.Decision Tree Classifier

Decision Tree Classifiers are generally effective for predicting binary labels, as they continually divide the dataset until all samples belonging to one class are isolated. However, one potential issue with this approach is overfitting, and thus we shall experiment with various hyperparameters to ensure that the model is generalizable for our dataset. Decisions Tree Classifier allows for us to make a generalization on many input values because each of the connecting nodes take in information to make an educated guess between each edge to get to our classification. In our model, we decided to tune the model's hyperparameters in order to optimize model performance. We tuned the criterion and maximum depth. Criterion measures the quality of each split and maximum depth controls the depth of our decision tree. We ran the list of parameters through GridSearchCV using 10 cross validation splits. The best values were gini and 14 for criterion, and maximum depth respectively. After training each split, we noticed that there were more samples being predicted incorrectly within the 70-30 split and 50-50 split. This is a major problem because we want to make sure that the model is accurately predicting hotel cancellations based on the confusion matrix. So, our best split for this model would be 80-20.

## B.Perceptron

Perceptron models can have binary classifications because the data is separated into linearly defined patterns to process the data. The Perceptron is a supervised learning model to take the input values and multiply them with the weights which will be added to decide whether the outcome will become a 0 or a 1. It works by finding the hyperplane that separates the two classes in the feature space. As the weights are updated for each of the features based on the misclassification errors. The result is a decision boundary that separates the two classes that can then predict weather our

output data is within a specific class. We also decided to use different hyperparameters as well in this model. We tuned eta0 in order to control the learning rate of our model which will change the weights of the model during each iteration. We GridSearchCV to cross validate across 10 folds and decided to find the best combination of each split. All test sizes determined that 1000 was the best iterations for the model but the learning rate was different for each best version of the model. The 80-20 split did better with a very small learning rate of .01, 70-30 experienced a learning rate of .1, while the 50-50 split experienced the highest learning rate of 1. Since each learning rate was vastly different the model trend lines between test and validation varied. Only one split shows promising results as the test and validation set lines start to almost converge within the 80-20 split while there are still massive gaps within the trend lines of the other split lines.

## C.Gaussian Naïve Bayes

Gaussian Naïve Bayes is mostly based on the assumption of the data being mostly independent and being normally distributed. Naïve Bayes also is supported by the Bayes Theorem which takes the probability of the classes given the features of the data which allows us to create a mean and variance. For our pipeline we decided to create a new Column Transformer for each split to include our one hot vector encoding in conjunction to our categorical data and scaling numerical variables. Overall, our results show that there is a similarity within the splits specifically when classifying if a hotel is not cancelled, our model predicts that those hotels are actually cancelled which is a huge problem. As the split size decreases there are more and more predicted wrong labels in this category; however, one split does better than the else. Our 70-30 split lines come super close to touch showing that our model may be overfitting, but this split contains better information since the test and validation lines are much closer than any other split.

## D.Logistic Regression

First, we think that Logistic Regression would be the best for binary classification because the model is great for nonlinear features and outcomes allowing for our model to be more flexible in predicting the cancelation of hotel rooms. We also decided to use more hyperparameters to evaluate out model penalty, solver, and max\_iter to switch through. Our penalty is the difference between L1 regularization versus L2 regularization in which changes the terms of our coefficients. L1 will take the absolute value of the coefficients while L2 as a the penalty term of the absolute value of the coefficients. Next, we used a solver in which we used to create optimization either with lbfgs which is a limited memory algorithm or using sag which is stochastic gradient descent. Lastly, we decided to add iterations ranging from 1000 to 2000 in increments of 500. For each of the splits, they all had hyperparameters that are the same such as 1500 iterations with l2 regularization and using stochastic gradient descent. In order to find the split that is best for the model we used the metrics of our f1-score over accuracy because we want to see whether or not our model is overfitting our data or not. One way we determined that 70-30 was our best split was first by looking at the learning curves for this model. The 70-30 split has a higher error within the validation set than any other split. To make sure that we aren't just picking the split based on this one assumption, when looking at the table metrics we see that the f1-score is much higher by at least 2 tenths than the other splits.

## E.Linear Regression

Linear Regression is used to find the line of best fit in terms of finding the relationship between our features to give an output that will predict a linear relationship between them all. Based on the learning curves for each split there are different ways in which the lines try to converge. When we look strictly at the 80-20 split, we can see that the two trend lines get closer to generalizing than any other split but if we look at the 70-30 split the lines follow a similar slope which usually is better for a model. Since the trend lines are not the best for picking the best model, it would be better to look for the mean standard error or even the R2 scores to see how the model generalizes using the variance of our variables. In contrast the 80-20 split still seems more of a viable solution for best model in terms of mean standard error with the highest score of .137 showing that there are less errors than the other splits but the R2 score is not as high as the 70-30 split. So in terms of fitting to the model the 70-30 split would be the best since it is most likely to handle newer data better than the 80-20 split.

## F.Ridge Rigression

Ridge Regression differs from linear regression because not all of the input variables are weighted the same so Ridge Regression allows for there to be shrinking coefficients so that some variables are penalized. Each penalty allows for there to be less overfitting within the models training data and keeps highly correlated variables. This is why Ridge regression is linear regression with regularization in order to better fit the model. For this model we decided to use different alphas to see if one would be better for different splits. We decided to use alphas ranging from .001, .01, .1, 1, 10, and 100 in conjunction with 10 cross validation folds. For our model most of our splits do best with an alpha of 1 but our 50-50 split is better with an alpha of 10; however, even though different alphas were better for each split the splits still follow a similar pattern as linear regression. Specifically the best model split is still 70-30 because each of the splits increases in the same way. The table between the two even shows that there is a resemblance but that the ridge regression model is better at not overfitting our data.

## G.Support Vector Machine (Linear Kernel)

The Support Vector Classifier is a straightforward algorithm that locates the optimal hyper plane of division between binary labels by maximizing the geometric margin and optimizing the hinge loss function. Three models were trained using this algorithm with a linear kernel on the different sample splits. No hyperparameter tuning was done to reduce runtime. The default parameters were used to train the models. All models generalize very well and exhibit low bias and low variance.

## H.Support Vector Machine (RBF Kernel)

Three additional models were trained utilizing support vector machines. The kernel specified for the model was the Radial Basis Function (RBF). The RBF Kernel on two samples x and x' is defined as , where γ, known as parameter gamma, must be greater than 0. It is important to note that the most significant parameters when training an SVM with RBF Kernel are C and gamma. The parameter C controls the regularization of our model, while gamma controls the importance of a training sample during classification. One method to determine the optimal values for these parameters is through the use of Grid Search cross-validation for hyperparameter tuning. However, due to the computational expense of this method on the SVM algorithm for our dataset, we opted to manually select the values for the parameters and run the model through a 10-fold cross-validation. The value chosen for C was 1.0, and the value for gamma was scale, which is calculated as where X is a sample in the inputs. Fig. 8 shows the learning curves for model training over the three different samples. All three samples analyzed by the classifier demonstrate a notable characteristic of low bias and low variance. It is worth noting that both the training and validation errors are low, and they exhibit a similarity that is within a difference of approximately 2%. Based on these observations, it can be concluded that the algorithm generalizes well on our dataset.

## I.Gradient Boosting Classifier

The Gradient Boosting algorithm is a member of the ensemble family of boosting algorithms. It is designed to combine weak classifiers in order to create a strong classifier. The algorithm iteratively performs this process by improving each weak model by focusing on observations that the previous model misclassified. The end result is a strong classifier that is a combination of all the weak classifiers. The weak classifier used in this algorithm is a decision tree with a maximum depth of 3, also known as a decision stump. In this study, three models were trained using the Gradient Boosting algorithm on all three samples. To determine the best values for the parameters, a grid search was performed with a cross-validation of 10 on the training data. The tuned parameters were criterion, loss, and the number of estimators. Criterion is a measure used to evaluate the quality of the split on the decision trees. Loss controls the type of function the model should optimize. The number of estimators refers to the number of iterations to perform in order to create the strong classifier. The best parameters for all sample splits were determined to be Friedman mean squared error, log loss, and 300 for criterion, loss, and number of estimators respectively. The learning curves on all samples are displayed in Fig. 9. All 3 models exhibit low bias, and low variance and generalize well on our dataset. Overall, the Gradient Boosting algorithm is a powerful tool for creating strong classifiers from weak classifiers. The use of decision trees with a maximum depth of 3, or decision stumps, is an effective way to create these weak classifiers. By tuning the parameters through a grid search with cross-validation, we were able to optimize the performance of the algorithm.

## J.Multi-layer Perceptron Classifier

The Multi-layer Perceptron (MLP) Classifier is a neural network that comprises multiple layers of neurons. Each layer is connected to the next layer by a set of weights. The MLP consists of an input layer, hidden layers, and output layer, which classify each input sample. The perceptron's neurons apply an activation function, such as the sigmoid or ReLU functions, to a weighted sum of the inputs. During training, the back propagation algorithm is used to learn the weights that produce an output. The back propagation algorithm utilizes a solver, which is an optimization algorithm that updates the weights of each neuron in the MLP model. Therefore, the solver, size of hidden layers, type of activation function, and the maximum number of iterations until convergence are some of the most important parameters. The values for the parameters in question were deliberately selected in order to minimize the runtime of the process. Specifically, the values chosen were the rectified linear unit (ReLU) function, the adam optimizer, 100 neurons in a single hidden layer, and 200 for the number of epochs. Three distinct models were trained using these parameters, and the learning curves depicted in Fig. 10 were obtained through a 10-fold cross validation on the input data. The learning curves themselves indicate that all three models exhibit low bias and high variance, which in turn results in a significant gap between the training and validation errors. Unfortunately, this means that the models are not able to properly generalize on our dataset and are instead overfitting.

## K.Random Forest Classifier

The Random Forest Classifier is an algorithm that belongs to the bagging ensemble family. It is designed to fit multiple decision trees on different subsets of the data and predict the class with the most votes. This approach is particularly useful in mitigating overfitting, as individual decision trees tend to generalize poorly on high-dimensional data. The main hyperparameters of the algorithm are the number of trees in the forest, the criterion, and the maximum depth of the tree. To optimize performance, these hyperparameters were tuned for all three models. The optimal number of estimators for the 50-50 split was found to be 100, while 200 was optimal for the other two models. The best value for the maximum depth of the tree was determined to be 18. In terms of the criterion, entropy was found to be the best choice for the 80-20 split, while gini was the optimal criterion for the other two models. The models underwent training on the input data utilizing their optimal parameters, which were obtained through grid search. As illustrated in Fig. 11, the learning curves for all three models are displayed. The first two models exhibit low bias and low variance, and they generalize well on the dataset. Conversely, the last model (50-50 split) has a low bias but high variance, and it fails to generalize to a certain degree. However, this outcome is to be expected due to the reduced training size for that particular model. On average, the algorithm generalizes on the dataset provided.

## L.K-Nearest Neighbors Classifier

The K-Nearest Neighbors (KNN) classifier is a non-parametric and lazy learning algorithm. It is commonly referred to as a lazy learner due to the fact that the generalization phase is postponed until a query is made to the model. Additionally, it is non-parametric in nature, as no assumptions are made on the distribution of the underlying dataset. As a result, it has no training phase and simply stores the training samples in memory. During the classification of new data points, the algorithm computes the distance between the new sample and the stored training samples. A prediction is made by voting among the k-nearest neighbors to the example being classified. The most important parameters for this algorithm are the number of neighbors, k, the metric used for distance calculation, and the weights used to vote on a label. These parameters were tuned using grid search with a 10-fold cross-validation. For each of the three models, the optimal values for metric, k, and weight were manhattan, 9, and distance respectively. The value, distance, signifies that the nearest neighbors were weighted by the inverse of their distance to the sample being classified. The learning curves of all three trained models are depicted in Fig. 12. It is important to mention that the training error for all three models is zero, as there is no training phase for the algorithm. The validation errors appear to be consistent across all three models for different sample sizes. Based on the available information, it can be concluded that the models seem to generalize properly on the dataset provided.

## M.AdaBoost Classifier

The AdaBoost Classifier is a boosting algorithm that iteratively trains weak classifiers by utilizing weights on the training data. During each iteration, a new weak classifier is trained on the dataset where the weights of previously misclassified samples are increased, while the weights of the correctly classified examples are decreased. This approach helps to increase the importance of the misclassified samples. It is worth noting that the weak classifier could be any weak learning classification algorithm. For the models trained, decision stumps were decided upon as the weak classifiers. The only parameter tuned by grid search during training was the number of estimators where boosting will be stopped. The best values were 200, 150, 100 for the 80-20, 70-30, and 50-50 splits respectively. The learning curves displayed in Fig. 13 show that all models have a low bias and a low variance. The similarity between the training and validation errors displays the algorithm's ability to generalize on the dataset.

# V.Model Evaluation and Interpretation

In the previous stage, a total of 39 top-performing models were selected for each of the 13 algorithms. To determine the effectiveness of these models on new and unseen data, they were subjected to evaluation on their respective test sets. The metrics utilized in this study to evaluate the models include accuracy, precision, recall, and f1-score. Accuracy is a measure of the ratio of correct predictions to the total number of predictions made. Precision, on the other hand, is the proportion of true positive predictions to the total positive predictions made. Conversely, recall is the ratio of true positives to the total actual positive samples. In the context of our study, the positive class pertains to the class of cancelled bookings. F1-score, which is the harmonic mean of precision and recall, gives equal weight to both metrics. Given the mild imbalance of our dataset, the F1-score is the most critical metric to consider as it provides a balance between precision and recall.

The metrics obtained from the three models of each algorithm were found to be very similar to one another, with most falling within a range of approximately 0.01. It was observed that for each classification algorithm, the model with the highest f1-score was chosen as the best model. On the other hand, for the regression algorithms, the R-squared (R2) score was deemed to be the decisive metric. The R2 score is a statistical measure that indicates the degree to which the input data fits the regression model. The evaluation metrics for each algorithm and sample split are presented in Tables 1,2, and 3. In the ensuing section, the top 13 models and their evaluation metrics are discussed, and the best overall performing model is determined.

# VI.Discussion and Conlusion

Table 4 illustrates the top 13 models which were determined by the highest f1-score for each classifier. From all 13 models, the best overall performing model is Gradient Boosting. This was decided as the best because it is the model with the highest f1-score with a value 0f 0.8016. The model has an accuracy of 0.8608, a precision of 0.8431 and recall of 0.764. It is also the model which generalizes the best, with a generalization error of approximately 0.02. The different samplings made no difference in the models for each classifier. All numbers for each algorithm were very similar to one another. However, it seems that for all algorithms, the model trained on the 50-50 split had the lowest metrics. This is most likely attributed to the fact that the model was trained on smaller training size compared to the other two samples. To improve model performance, we could have performed feature selection on our dataset before training. Selecting only the most important features for predicting cancellation would help reduce the dimension and increase model performance. We could have tuned more hyperparameters during model training. Although this increases runtime quadratically, it would have slightly improvements in performance.

##### References

1. Gawali, Suvarna. "End-to-End Hotel Booking Cancellation Machine Learning Model." _Analytics Vidhya_, 29 Mar. 2022, [https://www.analyticsvidhya.com/blog/2022/03/end-to-end-hotel-booking-cancellation-machine-learning-model/](https://www.analyticsvidhya.com/blog/2022/03/end-to-end-hotel-booking-cancellation-machine-learning-model/).
2. Jesse Mostipak. Febuary 2020. Hotel booking demand, Version 1. Retrieved Mar. 2022 from [Hotel booking demand | Kaggle](https://www.kaggle.com/datasets/jessemostipak/hotel-booking-demand).
3. N. Antonio, A. de Almeida, and L. Nunes, "Predicting hotel booking cancellations to decrease uncertainty and increase revenue," _Tourism & Management Studies_, vol. 13, no. 2, pp. 25-39, 2017.
4. N. A. Putro, R. Septian, W. Widiastuti, M. Maulidah, and H. F. Pardede, "Prediction of Hotel Booking Cancellation using Deep Neural Network and Logistic Regression Algorithm," _Jurnal Techno Nusa Mandiri_, vol. 18, no. 1, pp. 1–8, Mar. 2021, doi: [https://doi.org/10.33480/techno.v18i1.2056](https://doi.org/10.33480/techno.v18i1.2056).

**conference temcontain guidance text for your paper not being published**

Figure 1: splits for Decision Tree Classifier
![image](https://user-images.githubusercontent.com/77362614/234162631-7842498b-0282-4f0d-9bb7-f9449d533dba.png)

Figure 2: splits for Perceptron
![image](https://user-images.githubusercontent.com/77362614/234162738-e108a768-2b27-408c-9d0e-eae6cf2fbf33.png)

Figure 3: splits for Gaussian Naïve Bayes
![image](https://user-images.githubusercontent.com/77362614/234162787-8b57d132-2367-40ad-b3e4-407418812bb7.png)

Figure 4: splits for Logistic Regression
![image](https://user-images.githubusercontent.com/77362614/234162834-0c631ce3-0a87-4953-b991-42c2f5a5c644.png)

Figure 5: splits for Linear Regression
![image](https://user-images.githubusercontent.com/77362614/234162856-3b3d1ed8-644e-4319-9a63-7b12c59db3da.png)

Figure 6: splits for Ridge Regression
![image](https://user-images.githubusercontent.com/77362614/234162944-748dc7bf-5fb7-4f46-835a-547ef55c4510.png)

Figure 7: splits for (SVC Linear Kernel)
![image](https://user-images.githubusercontent.com/77362614/234162984-d3ba5936-0d91-48c4-96a9-6feb32f3b3a5.png)

Figure 8: splits for SVC (RBF Kernel)
![image](https://user-images.githubusercontent.com/77362614/234163047-e303ebf4-4ee3-42b0-8516-329c25cf94fe.png)

Figure 9: splits for Gradient Boosting Classifier
![image](https://user-images.githubusercontent.com/77362614/234163137-ea6fbdb2-b2a1-4f5c-b7df-59462ddc64d9.png)

Figure 10: splits for MLP Classifier
![image](https://user-images.githubusercontent.com/77362614/234163154-7be47c42-dc2c-4e23-a54f-5f14f9045b74.png)

Figure 11: splits for Random Forest Classifier
![image](https://user-images.githubusercontent.com/77362614/234163185-93122edd-ae5c-427e-a195-bc06874f2b7d.png)

Figure 12: splits for K-Neighbors Classifier
![image](https://user-images.githubusercontent.com/77362614/234163218-09c675a6-7053-4ef3-874d-0e41f4a79a99.png)

Figure 13: splits for Ada Boost Classifier
![image](https://user-images.githubusercontent.com/77362614/234163237-5bdf376f-8024-4f0b-96bb-c6161fd4662a.png)


Table 1: 80-20 split table for all models

| **Algorithm** | **TP** | **FP** | **FN** | **TN** | **Precision** | **Recall** | **F1-Score** | **Accuracy** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | 1280 | 458 | 325 | 2687 | 0.7975 | 0.7365 | 0.7658 | 0.8352 |
| Perceptron | 1249 | 489 | 527 | 2485 | 0.7033 | 0.7186 | 0.7109 | 0.7861 |
| Gaussian NB | 1717 | 21 | 2560 | 452 | 0.4014 | 0.9879 | 0.5709 | 0.4566 |
| Logistic Regression | 1148 | 590 | 257 | 2755 | 0.8171 | 0.6605 | 0.7305 | 0.8217 |
| SVM (Linear Kernel) | 1125 | 613 | 238 | 2774 | 0.8254 | 0.6473 | 0.7256 | 0.8208 |
| SVM (RBF Kernel) | 1219 | 519 | 210 | 2802 | 0.8530 | 0.7014 | 0.7698 | 0.8465 |
| Gradient Boosting | 1310 | 428 | 260 | 2752 | 0.8344 | 0.7537 | 0.7920 | 0.8552 |
| MLP | 1389 | 349 | 390 | 2622 | 0.7808 | 0.7992 | 0.7899 | 0.8444 |
| Random Forest | 1223 | 515 | 138 | 2874 | 0.8986 | 0.7037 | 0.7893 | 0.8625 |
| KNN | 1278 | 460 | 352 | 2660 | 0.7840 | 0.7353 | 0.7589 | 0.8291 |
| AdaBoost | 1232 | 506 | 254 | 2758 | 0.8291 | 0.7089 | 0.7643 | 0.8400 |

| **Algorithm** | **R2** | **Mean Squared Error** | **Mean Absolute Error** | **Median Absolute Error** |
| --- | --- | --- | --- | --- |
| Linear Regression | 0.4094 | 0.1370 | 0.3003 | 0.2607 |
| Ridge Regression | 0.4103 | 0.1368 | 0.3003 | 0.2599 |

Table 2: 70-30 split table for all models

| **Algorithm** | **TP** | **FP** | **FN** | **TN** | **Precision** | **Recall** | **F1-Score** | **Accuracy** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | 1955 | 668 | 445 | 4056 | 0.8146 | 0.7453 | 0.7784 | 0.8438 |
| Perceptron | 1614 | 1009 | 428 | 4073 | 0.7904 | 0.6153 | 0.6920 | 0.7983 |
| Gaussian NB | 2584 | 39 | 3798 | 703 | 0.4049 | 0.9851 | 0.5739 | 0.4614 |
| Logistic Regression | 1767 | 856 | 396 | 4105 | 0.8169 | 0.6737 | 0.7384 | 0.8243 |
| SVM (Linear Kernel) | 1723 | 900 | 351 | 4150 | 0.8308 | 0.6569 | 0.7337 | 0.8244 |
| SVM (RBF Kernel) | 1863 | 760 | 307 | 4194 | 0.8585 | 0.7103 | 0.7774 | 0.8502 |
| Gradient Boosting | 2004 | 619 | 373 | 4128 | 0.8431 | 0.7640 | 0.8016 | 0.8608 |
| MLP | 1972 | 651 | 471 | 4030 | 0.8072 | 0.7518 | 0.7785 | 0.8425 |
| Random Forest | 1835 | 788 | 191 | 4310 | 0.9057 | 0.6996 | 0.7894 | 0.8626 |
| KNN | 1926 | 697 | 545 | 3956 | 0.7794 | 0.7343 | 0.7562 | 0.8257 |
| AdaBoost | 1869 | 754 | 396 | 4105 | 0.8252 | 0.7125 | 0.7647 | 0.8386 |

| **Algorithm** | **R2** | **Mean Squared Error** | **Mean Absolute Error** | **Median Absolute Error** |
| --- | --- | --- | --- | --- |
| Linear Regression | 0.4225 | 0.1344 | 0.2980 | 0.2598 |
| Ridge Regression | 0.4234 | 0.1341 | 0.2980 | 0.2604 |

Table 3: 50-30 split table for all models

| **Algorithm** | **TP** | **FP** | **FN** | **TN** | **Precision** | **Recall** | **F1-Score** | **Accuracy** |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Decision Tree | 3234 | 1104 | 859 | 6676 | 0.7901 | 0.7455 | 0.7672 | 0.8347 |
| Perceptron | 2922 | 1416 | 1721 | 5814 | 0.6293 | 0.6736 | 0.6507 | 0.7358 |
| Gaussian NB | 4272 | 66 | 6395 | 1140 | 0.4005 | 0.9848 | 0.5694 | 0.4558 |
| Logistic Regression | 2878 | 1460 | 697 | 6838 | 0.8050 | 0.6634 | 0.7274 | 0.8183 |
| SVM (Linear Kernel) | 2843 | 1495 | 633 | 6902 | 0.8179 | 0.6554 | 0.7277 | 0.8208 |
| SVM (RBF Kernel) | 3050 | 1288 | 551 | 6984 | 0.8470 | 0.7031 | 0.7684 | 0.8451 |
| Gradient Boosting | 3278 | 1060 | 626 | 6909 | 0.8397 | 0.7556 | 0.7954 | 0.8580 |
| MLP | 3313 | 1025 | 897 | 6638 | 0.7869 | 0.7637 | 0.7752 | 0.8381 |
| Random Forest | 3014 | 1324 | 350 | 7185 | 0.8960 | 0.6948 | 0.7827 | 0.8590 |
| KNN | 3115 | 1223 | 900 | 6635 | 0.7758 | 0.7181 | 0.7458 | 0.8212 |
| AdaBoost | 3072 | 1266 | 686 | 6849 | 0.8175 | 0.7082 | 0.7589 | 0.8356 |

| **Algorithm** | **R2** | **Mean Squared Error** | **Mean Absolute Error** | **Median Absolute Error** |
| --- | --- | --- | --- | --- |
| Linear Regression | 0.4126 | 0.1362 | 0.3003 | 0.2646 |
| Ridge Regression | 0.4139 | 0.1359 | 0.3008 | 0.2671 |

Table 4: Best 13 Models

| **Algorithm** | **Precision** | **Recall** | **F1-Score** | **Accuracy** |
| --- | --- | --- | --- | --- |
| Decision Tree | 0.7975 | 0.7365 | 0.7658 | 0.8352 |
| Perceptron | 0.7033 | 0.7186 | 0.7109 | 0.7861 |
| Gaussian NB | 0.4049 | 0.9851 | 0.5739 | 0.4614 |
| Logistic Regression | 0.8169 | 0.6737 | 0.7384 | 0.8243 |
| SVM (Linear Kernel) | 0.8308 | 0.6569 | 0.7337 | 0.8244 |
| SVM (RBF Kernel) | 0.8585 | 0.7103 | 0.7774 | 0.8502 |
| Gradient Boosting | 0.8431 | 0.7640 | 0.8016 | 0.8608 |
| MLP | 0.7808 | 0.7992 | 0.7899 | 0.8444 |
| Random Forest | 0.9057 | 0.6996 | 0.7894 | 0.8626 |
| KNN | 0.7840 | 0.7353 | 0.7589 | 0.8291 |
| AdaBoost | 0.8291 | 0.7089 | 0.7643 | 0.8400 |

| **Algorithm** | **R2** | **Mean Squared Error** | **Mean Absolute Error** | **Median Absolute Error** |
| --- | --- | --- | --- | --- |
| Linear Regression | 0.4225 | 0.1344 | 0.2980 | 0.2598 |
| Ridge Regression | 0.4234 | 0.1341 | 0.2980 | 0.2604 |

©2023 IEEE
