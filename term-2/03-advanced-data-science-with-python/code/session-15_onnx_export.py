"""Export a scikit-learn pipeline to ONNX, run it with ONNX Runtime, check identical predictions, save a model card."""
import json
import numpy as np
import onnxruntime as ort
from skl2onnx import to_onnx
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression

X, y = load_breast_cancer(return_X_y=True)
X = X.astype(np.float32)
X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=.25, random_state=0, stratify=y)
pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000)).fit(X_tr, y_tr)

onx = to_onnx(pipe, X_tr[:1], options={id(pipe[-1]): {"zipmap": False}}, target_opset=17)
with open("tumour_model.onnx", "wb") as fh:
    fh.write(onx.SerializeToString())
print(f"ONNX file size: {len(onx.SerializeToString()) / 1024:.1f} KB | inputs: {[i.name for i in onx.graph.input]} "
      f"| outputs: {[o.name for o in onx.graph.output]}")
print("operators used:", sorted({node.op_type for node in onx.graph.node}))

sess = ort.InferenceSession("tumour_model.onnx", providers=["CPUExecutionProvider"])
labels, probs = sess.run(None, {sess.get_inputs()[0].name: X_te})
same_labels = np.array_equal(labels, pipe.predict(X_te))
max_diff = np.abs(probs - pipe.predict_proba(X_te)).max()
print(f"ONNX Runtime vs scikit-learn: identical labels = {same_labels}, max probability difference = {max_diff:.2e}")

card = {
    "model_name": "tumour-malignancy-logreg", "version": "1.0.0", "format": "ONNX opset 17",
    "intended_use": "decision support for clinicians reviewing biopsy measurements; not a diagnosis",
    "training_data": "Wisconsin breast cancer measurements (569 cases, 30 features)",
    "metrics": {"test_accuracy": round(float((labels == y_te).mean()), 3)},
    "limitations": ["trained on one hospital's data", "not validated on other populations"],
    "owner": "data science team", "explainability": "coefficients and SHAP values available"}
with open("model_card.json", "w") as fh:
    json.dump(card, fh, indent=2)
print("\nmodel card saved:", json.dumps(card)[:160], "...")
