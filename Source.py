from numpy import mean,array
from sklearn.linear_model import SGDRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split


#load Data
from sklearn.datasets import fetch_california_housing
Data = fetch_california_housing()
X = array(Data.data)
Y = array(Data.target)
Labels = Data.feature_names + Data.target_names


# Split Data & Scalling Data
Scaler = StandardScaler()
X_Train,X_Test,Y_Train,Y_Test = train_test_split(X,Y,test_size=0.3,random_state=4)
X_Train = Scaler.fit_transform(X_Train)
X_Test = Scaler.transform(X_Test)


#Building Model
SGDReg= SGDRegressor(alpha=0.9,shuffle=False)
SGDReg.fit(X_Train,Y_Train)
Result = SGDReg.predict(X_Test)
print(f'Mean Square Error is : {mean((Result - Y_Test) ** 2)}')