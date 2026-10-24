from sklearn.metrics import confusion_matrix, classification_report, cohen_kappa_score

actual    = ["yes"] * 9 + ["no"] * 5
predicted = ["yes", "yes", "yes", "yes", "yes", "yes", "yes", "no", "no",
             "yes", "no", "no", "no", "yes"]
print("Confusion matrix (rows = actual yes/no):")
print(confusion_matrix(actual, predicted, labels=["yes", "no"]))
print()
print(classification_report(actual, predicted, labels=["yes", "no"], digits=3))
print("Kappa:", round(cohen_kappa_score(actual, predicted), 3))
