import sys
import cv2
import numpy as np
import pyqtgraph as pg
from PyQt5.QtWidgets import QApplication, QMainWindow, QPushButton, QVBoxLayout, QWidget, QSlider, QLabel, QFileDialog, QHBoxLayout, QComboBox, QTextEdit, QSplitter, QLineEdit, QRadioButton, QProgressBar
from PyQt5.QtCore import Qt, QTimer, QThread, pyqtSignal
from PyQt5.QtGui import QImage, QPixmap, QIcon
import json
import time
import os
from ultralytics import YOLO



# https://stackoverflow.com/questions/31836104/pyinstaller-and-onefile-how-to-include-an-image-in-the-exe-file

def resource_path(relative_path):
    """ Get absolute path to resource, works for dev and for PyInstaller """
    try:
        # PyInstaller creates a temp folder and stores path in _MEIPASS
        base_path = sys._MEIPASS2
    except Exception:
        base_path = os.path.abspath(".")

    return os.path.join(base_path, relative_path)





# Define class name mappings for different datasets (Here: COCO dataset)
class_name_mappings = {
    'COCO': {
        0: 'person', 1: 'bicycle', 2: 'car', 3: 'motorcycle', 4: 'airplane', 5: 'bus', 6: 'train', 7: 'truck', 8: 'boat', 
        9: 'traffic light', 10: 'fire hydrant', 11: 'stop sign', 12: 'parking meter', 13: 'bench', 14: 'bird', 15: 'cat', 
        16: 'dog', 17: 'horse', 18: 'sheep', 19: 'cow', 20: 'elephant', 21: 'bear', 22: 'zebra', 23: 'giraffe', 
        24: 'backpack', 25: 'umbrella', 26: 'handbag', 27: 'tie', 28: 'suitcase', 29: 'frisbee', 30: 'skis', 
        31: 'snowboard', 32: 'sports ball', 33: 'kite', 34: 'baseball bat', 35: 'baseball glove', 36: 'skateboard', 
        37: 'surfboard', 38: 'tennis racket', 39: 'bottle', 40: 'wine glass', 41: 'cup', 42: 'fork', 43: 'knife', 
        44: 'spoon', 45: 'bowl', 46: 'banana', 47: 'apple', 48: 'sandwich', 49: 'orange', 50: 'broccoli', 
        51: 'carrot', 52: 'hot dog', 53: 'pizza', 54: 'donut', 55: 'cake', 56: 'chair', 57: 'couch', 58: 'potted plant', 
        59: 'bed', 60: 'dining table', 61: 'toilet', 62: 'tv', 63: 'laptop', 64: 'mouse', 65: 'remote', 66: 'keyboard', 
        67: 'cell phone', 68: 'microwave', 69: 'oven', 70: 'toaster', 71: 'sink', 72: 'refrigerator', 73: 'book', 
        74: 'clock', 75: 'vase', 76: 'scissors', 77: 'teddy bear', 78: 'hair drier', 79: 'toothbrush'
    },
    
}

class VideoProcessor(QThread):
    progress = pyqtSignal(int)
    finished = pyqtSignal(str, str)

    def __init__(self, video_path, model_name):
        super().__init__()
        self.video_path = video_path
        self.model_name = model_name

    def run(self):
        base_name = os.path.splitext(os.path.basename(self.video_path))[0]
        output_video = f"{base_name}_annotated_{self.model_name}.mp4"
        output_json = f"{base_name}_detection_data_{self.model_name}.json"

        model = YOLO(self.model_name)
        cap = cv2.VideoCapture(self.video_path)
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_video, fourcc, 20.0, (int(cap.get(3)), int(cap.get(4))))

        data = []
        frame_num = 0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            results = model.track(frame, persist=True)
            frame_ = results[0].plot()
            out.write(frame_)

            for box in results[0].boxes:
                bbox = box.xywh.cpu().numpy()[0]
                if hasattr(box, 'id') and box.id is not None:
                    id_value = int(box.id.cpu().numpy()[0])
                else:
                    id_value = None
                data.append({
                    'frame': int(cap.get(cv2.CAP_PROP_POS_FRAMES)),
                    'class': int(box.cls.cpu().numpy()[0]),
                    'confidence': float(box.conf.cpu().numpy()[0]),
                    'id': id_value,
                    'x': float(bbox[0]),
                    'y': float(bbox[1]),
                    'width': float(bbox[2]),
                    'height': float(bbox[3])
                })

            frame_num += 1
            self.progress.emit(int((frame_num / total_frames) * 100))  

        cap.release()
        out.release()
        cv2.destroyAllWindows()

        with open(output_json, 'w') as f:
            json.dump(data, f)

        self.finished.emit(output_video, output_json)  

class VideoGraphSync(QMainWindow):
    def __init__(self):
        super().__init__()

        self.class_names = {} 
        self.initUI()
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_frame)
        self.cap = None
        self.frame_count = 0
        self.fps = 0
        self.duration = 0
        self.current_frame = 0
        self.frame_data = {}
        self.current_class = None
        self.current_id = None
        self.window_size = 200  
        self.is_playing = False
        self.last_slider_value = 0

    def initUI(self):
        """
        Initialize the User Interface components of the application.
        """
        self.setWindowTitle('Actifilm')

        # Set window icon
        icon_path = resource_path("actifilm_icon.icns")
        self.setWindowIcon(QIcon(icon_path))

        # Video display
        self.video_label = QLabel(self)
        self.video_label.setAlignment(Qt.AlignCenter)
        self.video_label.setMinimumSize(320, 240)  

        # Graph display
        self.graph_widget = pg.GraphicsLayoutWidget(show=True, title="Temporal Series Graphs")
        self.plot = self.graph_widget.addPlot(title="Confidence Levels over Time (x: time(s) ; y: confidence(0 to 1) )")
        self.plot.addLegend()

        # Play, Pause and Reset buttons
        self.play_button = QPushButton('Play', self)
        self.play_button.clicked.connect(self.play_video)

        self.pause_button = QPushButton('Pause', self)
        self.pause_button.clicked.connect(self.pause_video)

        self.reset_button = QPushButton('Reset', self)
        self.reset_button.clicked.connect(self.reset_video)

        # Slider
        self.slider = QSlider(Qt.Horizontal, self)
        self.slider.sliderMoved.connect(self.slider_moved)
        self.slider.sliderReleased.connect(self.slider_released)

        # Upload buttons
        self.upload_video_button = QPushButton('Upload Video for Processing', self)
        self.upload_video_button.clicked.connect(self.upload_video)

        self.upload_annotated_video_button = QPushButton('Upload Annotated Video', self)
        self.upload_annotated_video_button.clicked.connect(self.upload_annotated_video)

        self.upload_json_button = QPushButton('Upload JSON Data', self)
        self.upload_json_button.clicked.connect(self.upload_json)

        # Dataset/model selection
        self.model_selector = QComboBox(self)
        self.update_model_list()
        self.model_selector.currentIndexChanged.connect(self.update_model)

        # Class selection
        self.class_selector = QComboBox(self)
        self.class_selector.currentIndexChanged.connect(self.update_class)

        # Search bar for class names
        self.search_bar = QLineEdit(self)
        self.search_bar.setPlaceholderText("Search for a class name (e.g., 'book')")
        self.search_bar.textChanged.connect(self.filter_class_names)

        # Search results display
        self.search_results_display = QTextEdit(self)
        self.search_results_display.setReadOnly(True)
        self.search_results_display.setFixedHeight(100)

        # Mode selection
        self.processing_mode_button = QRadioButton('Processing Mode', self)
        self.processing_mode_button.setChecked(True)
        self.processing_mode_button.toggled.connect(self.update_mode)

        self.visualization_mode_button = QRadioButton('Visualization Mode', self)
        self.visualization_mode_button.toggled.connect(self.update_mode)

        # Timestamps display
        self.timestamps_display = QTextEdit(self)
        self.timestamps_display.setReadOnly(True)

        # Static labels
        coco_mappings_label = QLabel("COCO Dataset Mappings", self)
        timestamps_label = QLabel("Detected Timestamps (in seconds) for the Selected Class", self)

        # Upload status display
        self.upload_status_label = QLabel(self)
        self.upload_status_label.setAlignment(Qt.AlignCenter)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setVisible(False)

        # Horizontal splitter for video and graph
        video_graph_splitter = QSplitter(Qt.Horizontal)
        video_graph_splitter.addWidget(self.video_label)
        video_graph_splitter.addWidget(self.graph_widget)
        video_graph_splitter.setStretchFactor(0, 1)
        video_graph_splitter.setStretchFactor(1, 3)

        # Vertical splitter for the overall layout
        main_splitter = QSplitter(Qt.Vertical)
        main_splitter.addWidget(video_graph_splitter)
        main_splitter.addWidget(self.slider)
        main_splitter.addWidget(timestamps_label)
        main_splitter.addWidget(self.timestamps_display)
        main_splitter.addWidget(coco_mappings_label)
        main_splitter.addWidget(self.search_bar)
        main_splitter.addWidget(self.search_results_display)

        # Control layout
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.processing_mode_button)
        control_layout.addWidget(self.visualization_mode_button)
        control_layout.addWidget(QLabel("Model:"))
        control_layout.addWidget(self.model_selector)
        control_layout.addWidget(self.upload_video_button)
        control_layout.addWidget(self.upload_json_button)
        control_layout.addWidget(self.upload_annotated_video_button)
        control_layout.addWidget(self.play_button)
        control_layout.addWidget(self.pause_button)
        control_layout.addWidget(self.reset_button)
        control_layout.addWidget(QLabel("Class:"))
        control_layout.addWidget(self.class_selector)

        main_layout = QVBoxLayout()
        main_layout.addWidget(main_splitter, stretch=2)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.upload_status_label)
        main_layout.addWidget(self.progress_bar)

        central_widget = QWidget()
        central_widget.setLayout(main_layout)
        self.setCentralWidget(central_widget)
   
    def update_model_list(self):
        """
        Update the model selection dropdown with available YOLO models.
        """
        model_list = ['yolov8n.pt', 'yolov8s.pt', 'yolov8m.pt', 'yolov8l.pt', 'yolov8x.pt']
        self.model_selector.addItems(model_list)

    def update_model(self):
        """
        Update the class names mapping based on the selected model.
        """
        model = self.model_selector.currentText()
        self.class_names = class_name_mappings.get(model, {})
        print(f"Class names for model {model}: {self.class_names}") 
        self.update_class_selector()

    def upload_video(self):
        """
        Upload a video for processing and display its path.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Video for Processing", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)
        if file_name:
            self.video_path = file_name
            self.process_video()
            self.upload_status_label.setText(f"Video uploaded: {self.video_path}")
            print(f"Video uploaded: {self.video_path}")
        else:
            self.upload_status_label.setText("No video selected")

    def upload_annotated_video(self):
        """
        Upload an annotated video for visualization and display its path.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload Annotated Video", "", "Video Files (*.mp4 *.avi *.mov);;All Files (*)", options=options)
        if file_name:
            self.annotated_video_path = file_name
            self.cap = cv2.VideoCapture(self.annotated_video_path)
            self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
            self.fps = int(self.cap.get(cv2.CAP_PROP_FPS))
            self.duration = self.frame_count / self.fps
            self.slider.setRange(0, self.frame_count)
            self.current_frame = 0
            self.upload_status_label.setText(f"Annotated video uploaded: {self.annotated_video_path}")
            print(f"Annotated video uploaded: {self.annotated_video_path}")
        else:
            self.upload_status_label.setText("No annotated video selected")

    def process_video(self):
        """
        Process the uploaded video using the selected YOLO model.
        """
        model_name = self.model_selector.currentText()
        self.progress_bar.setVisible(True)  
        self.progress_bar.setValue(0)  

        self.video_processor = VideoProcessor(self.video_path, model_name)
        self.video_processor.progress.connect(self.progress_bar.setValue)
        self.video_processor.finished.connect(self.on_processing_finished)
        self.video_processor.start()


    def on_processing_finished(self, output_video, output_json):
        """
        Handles the output after video processing
        """
        self.progress_bar.setVisible(False)
        print(f"Output video saved to: {output_video}")
        print(f"Detection data saved to: {output_json}")

    def upload_json(self):
        """
        Upload a JSON data file and display its path.
        """
        options = QFileDialog.Options()
        file_name, _ = QFileDialog.getOpenFileName(self, "Upload JSON Data", "", "JSON Files (*.json);;All Files (*)", options=options)
        if file_name:
            self.data_file_path = file_name
            self.load_data()
            self.upload_status_label.setText(f"JSON data uploaded: {self.data_file_path}")
            print(f"JSON data uploaded: {self.data_file_path}")
        else:
            self.upload_status_label.setText("No JSON data selected")

    def load_data(self):
        """
        Load detection data from a JSON file into the application.
        """
        with open(self.data_file_path, 'r') as f:
            self.data = json.load(f)

        self.frame_data = {}
        self.classes = set()

        for entry in self.data:
            frame = entry['frame']
            cls = entry['class']

            if frame not in self.frame_data:
                self.frame_data[frame] = []
            self.frame_data[frame].append(entry)

            self.classes.add(cls)

        self.update_class_selector()

    def update_class_selector(self):
        """
        Update the class selection dropdown based on loaded detection data.
        """
        self.class_selector.clear()
        class_list = [self.class_names.get(c, f'Class {c}') for c in sorted(self.classes)]
        self.class_selector.addItems(class_list)
        self.current_class = sorted(self.classes)[0] if self.classes else None
        self.update_graph()

    def update_mode(self):
        """
        Update the UI elements based on the selected mode (Processing/Visualization).
        """
        if self.processing_mode_button.isChecked():
            self.play_button.setEnabled(False)
            self.pause_button.setEnabled(False)
            self.reset_button.setEnabled(False)
            self.upload_json_button.setEnabled(False)
            self.upload_annotated_video_button.setEnabled(False)
            self.upload_video_button.setEnabled(True)
            self.model_selector.setEnabled(True)
        else:
            self.play_button.setEnabled(True)
            self.pause_button.setEnabled(True)
            self.reset_button.setEnabled(True)
            self.upload_json_button.setEnabled(True)
            self.upload_annotated_video_button.setEnabled(True)
            self.upload_video_button.setEnabled(False)
            self.model_selector.setEnabled(False)

    def play_video(self):
        """
        Start playing the video.
        """
        if self.cap:
            self.timer.start(1000 // self.fps)
            self.is_playing = True

    def pause_video(self):
        """
        Pause the video playback.
        """
        self.timer.stop()
        self.is_playing = False

    def reset_video(self):
        """
        Reset the video to the beginning.
        """
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            self.current_frame = 0
            self.slider.setValue(self.current_frame)
            self.update_frame()

    def slider_moved(self, position):
        """
        Update the current frame based on the slider position.
        """
        if self.cap:
            self.current_frame = position

    def slider_released(self):
        """
        Update the video frame when the slider is released.
        """
        if self.cap:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.current_frame)
            self.update_frame()

    def update_class(self):
        """
        Update the current class based on the selected class from the dropdown.
        """
        selected_text = self.class_selector.currentText()
        print(f"Selected class text: {selected_text}")

        if selected_text.startswith('Class '):
            selected_class_number = int(selected_text.split(' ')[1])
            self.current_class = selected_class_number
        else:
            for key, value in self.class_names.items():
                if value == selected_text:
                    self.current_class = key
                    break
            else:
                self.current_class = None

        print(f"Current class key: {self.current_class}")  
        self.update_graph()

    def update_frame(self):
        """
        Update the video frame and refresh the graph.
        """
        if self.cap:
            ret, frame = self.cap.read()
            if not ret:
                self.timer.stop()
                return

            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
            self.slider.setValue(self.current_frame)

            timestamp = time.strftime('%H:%M:%S', time.gmtime(self.current_frame / self.fps))
            cv2.putText(frame, timestamp, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            h, w, ch = rgb_image.shape
            bytes_per_line = ch * w
            q_image = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
            q_image = q_image.scaled(self.video_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.video_label.setPixmap(QPixmap.fromImage(q_image))
            
            self.update_graph()

    def update_graph(self):
        """
        Update the graph with confidence levels of detections over time.
        """
        self.plot.clear()
        start_frame = max(0, self.current_frame - self.window_size // 2)
        end_frame = min(self.frame_count, self.current_frame + self.window_size // 2)

        time_series = []
        confidence_series = []

        scatter_time_series = []
        scatter_confidence_series = []
        
        class_name = self.class_names.get(self.current_class, f'Class {self.current_class}')

        for frame in range(start_frame, end_frame):
            if frame in self.frame_data:
                for entry in self.frame_data[frame]:
                    if entry['class'] == self.current_class:
                        time_series.append(frame / self.fps)
                        confidence_series.append(entry['confidence'])
                        scatter_time_series.append(frame / self.fps)
                        scatter_confidence_series.append(entry['confidence'])

        if time_series and confidence_series:
            self.plot.plot(time_series, confidence_series, pen='r', symbol='o', name=class_name)
        else:
            self.plot.plot([0], [0], pen=None, symbol='o', name="No data")

        self.update_timestamps_display()

    def update_timestamps_display(self):
        """
        Update the display showing the timestamps of detections for the selected class.
        """
        self.timestamps_display.clear()
        timestamps = [] 
        if self.current_class is not None:
            for frame, entries in self.frame_data.items():
                for entry in entries:
                    if entry['class'] == self.current_class:
                        timestamps.append(frame / self.fps)

        if timestamps:
            timestamps_text = "Detected Timestamps (in seconds) for the selected class:\n"
            timestamps_text += "\n".join([f"Detection at second {timestamp:.2f} " for timestamp in sorted(timestamps)])
            self.timestamps_display.setText(timestamps_text)
        else:
            self.timestamps_display.setText("No detections found for the selected class.")

    def get_coco_class_mappings_text(self):
        """
        Get a formatted text of COCO class mappings.
        """
        coco_class_mappings = class_name_mappings.get('COCO', {})
        class_mappings_text = "COCO Dataset Class Mappings:\n"
        class_mappings_text += "\n".join([f"Class {num}: {name}" for num, name in coco_class_mappings.items()])
        return class_mappings_text
    
    def filter_class_names(self, query):
        """Filter the COCO class mappings based on the search query and display the results."""
        coco_class_mappings = class_name_mappings.get('COCO', {})
        filtered_mappings = {num: name for num, name in coco_class_mappings.items() if query.lower() in name.lower()}

        if filtered_mappings:
            search_results_text = "\n".join([f"Class {num}: {name}" for num, name in filtered_mappings.items()])
        else:
            search_results_text = "No matching classes found."

        self.search_results_display.setText(search_results_text)

def main():
    """
    Main function to run the application.
    """
    app = QApplication(sys.argv)
    ex = VideoGraphSync()
    ex.show()
    sys.exit(app.exec_())

if __name__ == '__main__':
    main()

