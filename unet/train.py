import tensorflow as tf
import hyperparameters as hp


# IoU Loss
def iou_loss(y_true, y_pred):
    epsilon = 1e-7
    intersection = tf.reduce_sum(y_true * y_pred, axis=[1, 2, 3])
    union = tf.reduce_sum(y_true + y_pred, axis=[1, 2, 3]) - intersection
    return 1 - (intersection + epsilon) / (union + epsilon)

# Combined loss: binary_crossentropy + IoU loss
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    iou = iou_loss(y_true, y_pred)
    return bce + iou  

def train_model(model, train_dataset, val_dataset, epochs=hp.num_epochs, batch_size=hp.batch_size):
    callbacks = [
        tf.keras.callbacks.ModelCheckpoint(
            "best_model.h5", save_best_only=True, monitor="val_loss", mode="min"
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_loss", patience=10, restore_best_weights=True
        ),
        # decrease learning rate if loss stop reducing
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.1, patience=5, min_lr=1e-6
        )
    ]

    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp.learning_rate),
        loss= combined_loss,
        metrics=["accuracy", tf.keras.metrics.MeanIoU(num_classes=2)]
    )

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks
    )
    return history
