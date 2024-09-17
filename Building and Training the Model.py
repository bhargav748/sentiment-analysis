# Create a pipeline with a CountVectorizer and Multinomial Naive Bayes
pipeline = Pipeline([
    ('vect', CountVectorizer()),
    ('clf', MultinomialNB())
])

# Train the model
pipeline.fit(X_train, y_train)
