# NLP-Fake-News-Classification
Fake news Classification using Deep Learning,
Total data size 4049,
missing values 21,
Data has features URLs, Headline,Body; output Lebel,
List of models used:
Logistic regression-Accuracy 84.46% (kappa score= .68445)
RF Classifier-Accuracy 84.56%(kappa score= .6825)
XgBoost -Accuracy 79.19%(kappa score=.58)
CatBoost-Accuracy 81% (kappa Score=.61)
LSTM-Accuracy 85%(kappa score=.6412)
Bi-Directional LSTM- Accuracy 90%( kappa score=.7930)
I have also implemented other performence matric ( confusion matrix, classification report, roc_auc score) to check which model actually perfrom better.
For hyper-parameter optimization have used RandomizedSearchCV.
