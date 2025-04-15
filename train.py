# Train ensemble on your dataset
ensemble.fit(X_train, y_train)
# Save the trained ensemble
pickle.dump(ensemble, open("model/ensemble_model.pkl", "wb"))
