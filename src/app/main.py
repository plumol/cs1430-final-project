import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread, pyqtSignal, pyqtSlot, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import QApplication, QLabel, QMainWindow, QVBoxLayout, QWidget, QHBoxLayout, QPushButton, QFileDialog, QMessageBox
from unet import unet_model

# Thread class for capturing video
class VideoCaptureThread(QThread):
    frameCaptured = pyqtSignal(np.ndarray)
    newBackground = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.running = True

        self.capture = cv2.VideoCapture(0)  # 0 means the default camera
        self.frame_width = int(self.capture.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.capture.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.background_image = np.zeros((self.frame_height, self.frame_width, 3))

        self.input_size = 256
        self.unet = unet_model(input_shape=(self.input_size, self.input_size, 3))
        self.unet.load_weights("./model_path/best_model.h5")

        self.newBackground.connect(self.updateBackground)

    def updateBackground(self, background_path):
        if background_path == "":
            self.background_image = np.zeros((self.frame_height, self.frame_width, 3))
        else:
            self.background_image= cv2.imread(background_path)
            self.background_image = cv2.resize(self.background_image, (self.frame_width, self.frame_height))

    def run(self):
        while self.running:
            ret, frame = self.capture.read()
            if ret:
                # pre-processing
                img = cv2.resize(frame, (self.input_size, self.input_size)) / 255.0
                img = img.astype(np.float32)
                img = np.expand_dims(img, axis=0)

                # model inference
                mask = self.unet.predict(img, verbose=0)[0]
                mask = cv2.resize(mask, (self.frame_width, self.frame_height))

                # post-processing
                foreground = frame.copy()
                foreground[mask < 0.3] = 0

                background = self.background_image.copy()
                background[mask >= 0.3] = 0

                self.frameCaptured.emit(foreground+background)

    def stop(self):
        self.running = False
        self.wait()
        self.capture.release()

# Main Window class
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt Camera with Thread")

        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        
        self.layout = QHBoxLayout(self.central_widget)

        # video
        self.video_width = 640
        self.video_height = 480
        self.video_label = QLabel("Camera Feed")
        self.video_label.setFixedSize(self.video_width, self.video_height)

        # buttons
        self.button_layout = QVBoxLayout()
        self.is_recording = False
        self.record_button = QPushButton("Start Recording")
        self.select_background_button = QPushButton("Select Background")
        self.reset_background_button = QPushButton("Reset Background")

        self.record_button.clicked.connect(self.recording)
        self.select_background_button.clicked.connect(self.changeBackground)
        self.reset_background_button.clicked.connect(self.clearBackground)

        self.button_layout.addWidget(self.record_button)
        self.button_layout.addWidget(self.select_background_button)
        self.button_layout.addWidget(self.reset_background_button)

        self.record_button.setFixedSize(100, 25)
        self.select_background_button.setFixedSize(100, 25)
        self.reset_background_button.setFixedSize(100, 25)

        self.layout.addWidget(self.video_label)
        self.layout.addLayout(self.button_layout)

        # timer
        self.record_timer = QTimer()
        self.record_timer_step = 40
        self.record_timer.timeout.connect(self.writeFrame)

        # thread
        self.video_thread = VideoCaptureThread()
        self.video_thread.frameCaptured.connect(self.updateFrame)
        self.video_thread.start()

    @pyqtSlot(np.ndarray)
    def updateFrame(self, frame):
        frame = cv2.resize(frame.astype(np.uint8)[:, ::-1, :], (self.video_width, self.video_height))

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = rgb_frame.shape
        bytes_per_line = ch * w
        qImg = QImage(rgb_frame.data, w, h, bytes_per_line, QImage.Format_RGB888)

        self.video_label.setPixmap(QPixmap.fromImage(qImg))

    def recording(self):
        self.is_recording = not self.is_recording
        if self.is_recording:
            self.record_button.setText("Stop Recording")
            self.record_timer.start(self.record_timer_step)
            fourcc = cv2.VideoWriter_fourcc(*'XVID')
            self.out = cv2.VideoWriter('output.avi', fourcc, 1000/self.record_timer_step, (self.video_width, self.video_height))
            QMessageBox.information(self, "Info", "Start Recording!")
        else:
            self.record_button.setText("Start Recording")
            self.record_timer.stop()
            self.out.release()
            QMessageBox.information(self, "Info", "Recording saved!")

    def changeBackground(self):
        background_path, _ = QFileDialog.getOpenFileName(self, "Select Image", "", "Images (*.png *.xpm *.jpg *.bmp)")
        if background_path:
            self.video_thread.newBackground.emit(background_path)
    
    def clearBackground(self):
        self.video_thread.newBackground.emit("")

    def writeFrame(self):
        if self.is_recording:
            image = self.video_label.pixmap().toImage()

            width = image.width()
            height = image.height()
            ptr = image.bits()
            ptr.setsize(image.byteCount())
            img = np.array(ptr).reshape((height, width, 4))
            img = img[:, :, :3]
            img = cv2.resize(img, (self.video_width, self.video_height))
            self.out.write(img)

    def closeEvent(self, event):
        if self.is_recording:
            self.out.release()
        # Stop the video thread on close
        self.video_thread.stop()
        super().closeEvent(event)

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())

from PyQt5.QtWidgets import QFileDialog, QMessageBox
import cv2

