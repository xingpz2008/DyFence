def compute_top1_accuracy(predictions, labels):
    predictions = list(predictions)
    labels = list(labels)
    if len(predictions) != len(labels) or not labels:
        raise ValueError("predictions and labels must have the same non-zero length")
    correct = sum(int(prediction) == int(label) for prediction, label in zip(predictions, labels))
    return {"correct": correct, "total": len(labels), "percentage": 100.0 * correct / len(labels)}
