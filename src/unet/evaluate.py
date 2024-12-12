def evaluate_model(model, val_dataset):
    loss, accuracy, mean_iou = model.evaluate(val_dataset)
    print(f"Validation Loss: {loss}")
    print(f"Validation Accuracy: {accuracy}")
    print(f"Validation Mean IoU: {mean_iou}")
