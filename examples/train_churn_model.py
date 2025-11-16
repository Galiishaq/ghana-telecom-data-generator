import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

def train_churn_model():
    """
    This function loads the synthetic customer data, trains a churn prediction model,
    and evaluates its performance.
    """
    # 1. Load Data
    try:
        df = pd.read_csv('data/ghana_telecom_customers.csv')
    except FileNotFoundError:
        print("Error: The dataset 'data/ghana_telecom_customers.csv' was not found.")
        return

    print("Dataset loaded successfully.")
    print(f"Dataset shape: {df.shape}")

    # Drop non-predictive features
    # customer_id: just an identifier
    # customer_segment: this is the "ground truth" label we used to generate data, not a real feature
    # churn_risk_score: this is the score we used to calculate churn, would leak target info
    cols_to_drop = ['customer_id']
    if 'customer_segment' in df.columns:
        cols_to_drop.append('customer_segment')
    if 'churn_risk_score' in df.columns:
        cols_to_drop.append('churn_risk_score')

    df = df.drop(cols_to_drop, axis=1)

    # 2. Define Features (X) and Target (y)
    X = df.drop('churned', axis=1)
    y = df['churned']

    # Identify categorical and numerical features
    categorical_features = ['region', 'locality_type', 'subscription_type', 'device_type', 'night_data_user']
    numerical_features = [col for col in X.columns if col not in categorical_features]

    print("\nTarget variable distribution (churn):")
    print(y.value_counts(normalize=True))

    # 3. Preprocessing Pipeline
    # Create a column transformer to apply different transformations to different columns
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', StandardScaler(), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ],
        remainder='passthrough' # Keep other columns (if any)
    )

    # 4. Define the Model
    # Using RandomForestClassifier as requested
    model = RandomForestClassifier(random_state=42, n_estimators=100, class_weight='balanced')

    # 5. Create the full pipeline
    pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                               ('classifier', model)])

    # 6. Split Data into Training and Testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    print(f"\nTraining set size: {X_train.shape[0]}")
    print(f"Test set size: {X_test.shape[0]}")

    # 7. Train the Model
    print("\nTraining the RandomForestClassifier model...")
    pipeline.fit(X_train, y_train)
    print("Model training complete.")

    # 8. Evaluate the Model
    print("\nEvaluating model performance on the test set...")
    y_pred = pipeline.predict(X_test)

    # Print evaluation metrics
    print("\nAccuracy Score:")
    print(f"{accuracy_score(y_test, y_pred):.4f}")

    print("\nConfusion Matrix:")
    # Transposing for a more standard view:
    # [[TN, FP],
    #  [FN, TP]]
    cm = confusion_matrix(y_test, y_pred)
    print(cm)
    print(f"True Negatives (TN): {cm[0][0]} - Correctly predicted non-churners")
    print(f"False Positives (FP): {cm[0][1]} - Incorrectly predicted churners")
    print(f"False Negatives (FN): {cm[1][0]} - Incorrectly predicted non-churners (missed churns)")
    print(f"True Positives (TP): {cm[1][1]} - Correctly predicted churners")


    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['Not Churned (0)', 'Churned (1)']))

if __name__ == '__main__':
    train_churn_model()
