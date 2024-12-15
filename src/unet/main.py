from data_loader import generate_file_paths, create_dataset_pipeline
from models import unet_model
from train import train_model
from evaluate import evaluate_model
import hyperparameters as hp

image_dir = "../../data/images"
mask_dir = "../../data/masks"

# load data
image_paths, mask_paths = generate_file_paths(image_dir, mask_dir)

# split to validation and training data
from sklearn.model_selection import train_test_split

train_images, val_images, train_masks, val_masks = train_test_split(
    image_paths, mask_paths, test_size=0.2, random_state=42
)

# create dataset
train_dataset = create_dataset_pipeline(train_images, train_masks, batch_size= hp.batch_size)
val_dataset = create_dataset_pipeline(val_images, val_masks, batch_size= hp.batch_size)

# build model
model = unet_model(input_shape=(hp.img_size, hp.img_size, 3))

# see model structure
model.summary()

# training model
train_model(model, train_dataset, val_dataset, epochs=hp.num_epochs, batch_size= hp.batch_size)

# evaluate
model.load_weights("best_model.h5")
evaluate_model(model, val_dataset)
