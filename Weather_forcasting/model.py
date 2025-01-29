from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression 
import pandas as pd

def train_model(df):
    X = df[['temperature', 'humidity', 'windspeed']]  
    y = df['target_variable']  

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    model = LinearRegression() 
    model.fit(X_train, y_train)

    return model

def predict_weather(model, df):
    X_pred = df[['temperature', 'humidity', 'windspeed']]  
    predictions = model.predict(X_pred)
    predictions_df = pd.DataFrame(predictions, columns=['Predicted Weather'])
    return predictions_df