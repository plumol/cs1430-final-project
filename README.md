# Background Creator

This project implements a **U-Net model** for human segmentation tasks. The model is trained to segment human pixels from the background in input images.

## Dependencies

To set up the environment, you need conda cs1430 environment and the **PyQt5** package:

Install required packages, including **PyQt5**:

```bash
pip install PyQt5
```

## How to Run

### 1. Running the App (Mac Users)

Navigate to the `src` directory and run the following command to start the GUI-based segmentation app:

```bash
python3 main.py
```

### 2. Training the U-Net Model

Navigate to the `unet` directory and run:

```bash
python3 main.py
```

This will train the U-Net model from scratch.

### 3. Generating Mask Predictions

After training, you can produce mask predictions using your best model:

```bash
python3 predict.py
```

### 4. Running the App (Windows Users)

For Windows users, download the pre-trained model `best_model.h5` from [Google Drive](https://drive.google.com/file/d/1WPb1snFzzM_nzwEHY5uBYX_ZwO0wQdFC/view?usp=sharing). Place the downloaded `best_model.h5` in the **`app` directory**.

Run the app:

```bash
python3 main.py
```

## Dataset

We used the **Supervisely Filtered Segmentation Person Dataset** from Kaggle. You can download it here:

[https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset/data](https://www.kaggle.com/datasets/tapakah68/supervisely-filtered-segmentation-person-dataset/data)

## Demo Video

Watch the recorded demo video showcasing our project:

[https://www.youtube.com/watch?v=o7zNAWk1v5k](https://www.youtube.com/watch?v=o7zNAWk1v5k)

## References

We referenced the following resources while developing this project:

1. [Virtual Background for Video Conferencing](https://towardsdatascience.com/virtual-background-for-video-conferencing-using-machine-learning-dfba17d90aa9)
2. [U-Net Segmentation in TensorFlow](https://idiotdeveloper.com/unet-segmentation-in-tensorflow/)
3. [Virtual Background GitHub Repository](https://github.com/Volcomix/virtual-background?tab=readme-ov-file)
