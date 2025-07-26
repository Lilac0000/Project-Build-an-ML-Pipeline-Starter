# Add this section after loading the model (around line 75-80)

# Apply label encoders to categorical features before inference
for feature in model_data['categorical_features']:
    if feature in X_test.columns:
        if feature in model_data['label_encoders']:
            le = model_data['label_encoders'][feature]
            # Handle unseen categories by using the most frequent class or a default
            X_test[feature] = X_test[feature].map(
                lambda x: le.transform([x])[0] if x in le.classes_ else le.transform([le.classes_[0]])[0]
            )
        else:
            logger.warning(f"No label encoder found for categorical feature: {feature}")

# Ensure feature order matches training
X_test = X_test[expected_features]

# Convert to numeric (this should now work since categoricals are encoded)
X_test = X_test.select_dtypes(include=[np.number])

logger.info(f"Test data shape after preprocessing: {X_test.shape}")
logger.info(f"Test data dtypes: {X_test.dtypes}")

# Now perform inference
logger.info("Performing inference...")
y_pred = sk_pipe.predict(X_test)
