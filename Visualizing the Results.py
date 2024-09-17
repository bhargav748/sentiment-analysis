from sklearn.metrics import ConfusionMatrixDisplay

ConfusionMatrixDisplay.from_estimator(pipeline, X_test, y_test)
plt.title('Confusion Matrix')
plt.show()
