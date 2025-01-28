from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression  # Example model

def train_model(df):
    # Separate features (X) and target variable (y)
    X = df[['temperature', 'humidity', 'windspeed']]  # Example features
    y = df['target_variable']  # Replace with your target variable

    # Split data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train the model
    model = LinearRegression()  # Example model
    model.fit(X_train, y_train)

    return model

def predict_weather(model, df):
    # Prepare data for prediction
    X_pred = df[['temperature', 'humidity', 'windspeed']]  # Example features
    predictions = model.predict(X_pred)
    # Create a DataFrame for predictions
    predictions_df = pd.DataFrame(predictions, columns=['Predicted Weather'])
    return predictions_df