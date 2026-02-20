import mlflow
import mlflow.sklearn
from sklearn.datasets import load_wine
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score,confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Load wine Dataset
wine=load_wine()
X=wine.data
y=wine.target

# print(X)

# Train test Split
X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.3,random_state=1)

# Define the params for Random Forest Model
max_depth=10
n_estimators=15

with mlflow.start_run():
    rf=RandomForestClassifier(max_depth=max_depth,n_estimators=n_estimators,random_state=42)
    rf.fit(X_train,y_train)

    y_pred=rf.predict(X_test)
    accuracy=accuracy_score(y_test,y_pred)

    # Logging with MLFlow
    mlflow.log_metric('Accuracy',accuracy)
    mlflow.log_param('max_depth',max_depth)
    mlflow.log_param('n_estimators',n_estimators)

    # print(accuracy)

    # Creating a Confusion matrix plot
    cm=confusion_matrix(y_test,y_pred)
    plt.figure(figsize=(6,6))
    sns.heatmap(cm,annot=True,fmt='d',cmap='Blues',xticklabels=wine.target_names,yticklabels=wine.target_names)
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.title('Confusion Matrix')

    # Save plot
    plt.savefig("Confusion_Matrix.png")

    # Log Artifacts using mlflow
    mlflow.log_artifact('Confusion_Matrix.png')
    mlflow.log_artifact(__file__)

    print(accuracy)