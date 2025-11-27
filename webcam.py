# webcam.py
import cv2
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
from pathlib import Path

# ============================================================================
# MODEL DEFINITION (Same as training)
# ============================================================================

class DeepCNN(nn.Module):
    def __init__(self, num_classes=7):
        super(DeepCNN, self).__init__()

        # Block 1
        self.conv1_1 = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        self.bn1_1 = nn.BatchNorm2d(64)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, padding=1)
        self.bn1_2 = nn.BatchNorm2d(64)
        self.pool1 = nn.MaxPool2d(2, 2)
        self.dropout1 = nn.Dropout2d(0.4)

        # Block 2
        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
        self.bn2_1 = nn.BatchNorm2d(128)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, padding=1)
        self.bn2_2 = nn.BatchNorm2d(128)
        self.pool2 = nn.MaxPool2d(2, 2)
        self.dropout2 = nn.Dropout2d(0.4)

        # Block 3
        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, padding=1)
        self.bn3_1 = nn.BatchNorm2d(256)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, padding=1)
        self.bn3_2 = nn.BatchNorm2d(256)
        self.pool3 = nn.MaxPool2d(2, 2)
        self.dropout3 = nn.Dropout2d(0.5)

        # Block 4
        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, padding=1)
        self.bn4_1 = nn.BatchNorm2d(512)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, padding=1)
        self.bn4_2 = nn.BatchNorm2d(512)
        self.pool4 = nn.MaxPool2d(2, 2)
        self.dropout4 = nn.Dropout2d(0.5)

        # Global Average Pooling
        self.global_pool = nn.AdaptiveAvgPool2d(1)

        # Fully connected layers
        self.fc1 = nn.Linear(512, 256)
        self.bn_fc1 = nn.BatchNorm1d(256)
        self.dropout_fc1 = nn.Dropout(0.6)

        self.fc2 = nn.Linear(256, 128)
        self.bn_fc2 = nn.BatchNorm1d(128)
        self.dropout_fc2 = nn.Dropout(0.6)

        self.fc3 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Block 1: 48x48 -> 24x24
        x = F.relu(self.bn1_1(self.conv1_1(x)))
        x = F.relu(self.bn1_2(self.conv1_2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)

        # Block 2: 24x24 -> 12x12
        x = F.relu(self.bn2_1(self.conv2_1(x)))
        x = F.relu(self.bn2_2(self.conv2_2(x)))
        x = self.pool2(x)
        x = self.dropout2(x)

        # Block 3: 12x12 -> 6x6
        x = F.relu(self.bn3_1(self.conv3_1(x)))
        x = F.relu(self.bn3_2(self.conv3_2(x)))
        x = self.pool3(x)
        x = self.dropout3(x)

        # Block 4: 6x6 -> 3x3
        x = F.relu(self.bn4_1(self.conv4_1(x)))
        x = F.relu(self.bn4_2(self.conv4_2(x)))
        x = self.pool4(x)
        x = self.dropout4(x)

        # Global pooling: 3x3 -> 1x1
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)  # Flatten

        # FC layers
        x = F.relu(self.bn_fc1(self.fc1(x)))
        x = self.dropout_fc1(x)
        x = F.relu(self.bn_fc2(self.fc2(x)))
        x = self.dropout_fc2(x)
        x = self.fc3(x)

        return x

# ============================================================================
# WEBCAM APPLICATION
# ============================================================================

class EmotionDetector:
    def __init__(self, model_path='results/models/facial/facial_dcnn_best.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        # Emotion labels
        self.emotions = ['angry', 'disgust', 'fear', 'happy', 'neutral', 'sad', 'surprise']

        # Load model
        print("Loading model...")
        self.model = DeepCNN(num_classes=7).to(self.device)
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.eval()
        print(f"✓ Model loaded on {self.device}")

        # Image transforms (same as training)
        self.transform = transforms.Compose([
            transforms.Resize((48, 48)),
            transforms.Grayscale(num_output_channels=1),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5], std=[0.5])
        ])

        # Load face cascade for face detection
        cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
        self.face_cascade = cv2.CascadeClassifier(cascade_path)

        # Colors for each emotion
        self.colors = {
            'happy': (0, 255, 0),      # Green
            'neutral': (255, 255, 0),  # Yellow
            'surprise': (0, 255, 255), # Cyan
            'sad': (255, 0, 0),        # Blue
            'angry': (0, 0, 255),      # Red
            'disgust': (128, 0, 128),  # Purple
            'fear': (255, 165, 0)      # Orange
        }

    def preprocess_face(self, face_img):
        """Convert face image to tensor"""
        # Convert BGR to RGB
        face_rgb = cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB)

        # Convert to PIL Image
        pil_img = Image.fromarray(face_rgb)

        # Apply transforms
        tensor = self.transform(pil_img)

        # Add batch dimension
        tensor = tensor.unsqueeze(0)

        return tensor.to(self.device)

    def predict_emotion(self, face_img):
        """Predict emotion from face image"""
        tensor = self.preprocess_face(face_img)

        with torch.no_grad():
            output = self.model(tensor)
            probabilities = F.softmax(output, dim=1)
            confidence, predicted = torch.max(probabilities, 1)

        emotion = self.emotions[predicted.item()]
        conf = confidence.item() * 100

        return emotion, conf, probabilities[0].cpu().numpy()

    def run(self):
        """Start webcam emotion detection"""
        # Open webcam
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            print("Error: Cannot open webcam")
            return

        print("\n" + "="*60)
        print("REAL-TIME EMOTION DETECTION")
        print("="*60)
        print("Controls:")
        print("  'q' - Quit")
        print("  's' - Save screenshot")
        print("  'i' - Show/Hide info panel")
        print("="*60 + "\n")

        show_info = True
        frame_count = 0

        while True:
            ret, frame = cap.read()

            if not ret:
                print("Error: Cannot read frame")
                break

            frame_count += 1

            # Flip frame horizontally for mirror effect
            frame = cv2.flip(frame, 1)

            # Convert to grayscale for face detection
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

            # Detect faces
            faces = self.face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.1,
                minNeighbors=5,
                minSize=(48, 48)
            )

            # Process each detected face
            for (x, y, w, h) in faces:
                # Extract face region
                face_img = frame[y:y+h, x:x+w]

                # Predict emotion
                emotion, confidence, probs = self.predict_emotion(face_img)

                # Get color for this emotion
                color = self.colors.get(emotion, (255, 255, 255))

                # Draw rectangle around face
                cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)

                # Draw emotion label
                label = f"{emotion.upper()}: {confidence:.1f}%"
                cv2.putText(frame, label, (x, y-10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)

                # Draw info panel if enabled
                if show_info:
                    self.draw_info_panel(frame, probs)

            # Draw FPS
            cv2.putText(frame, f"Frame: {frame_count}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            # Show frame
            cv2.imshow('Emotion Detection - Press Q to quit', frame)

            # Handle key presses
            key = cv2.waitKey(1) & 0xFF

            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                filename = f'screenshot_{frame_count}.jpg'
                cv2.imwrite(filename, frame)
                print(f"✓ Screenshot saved: {filename}")
            elif key == ord('i'):
                show_info = not show_info

        # Cleanup
        cap.release()
        cv2.destroyAllWindows()
        print("Webcam closed.")

    def draw_info_panel(self, frame, probabilities):
        """Draw emotion probabilities panel"""
        h, w = frame.shape[:2]
        panel_width = 250
        panel_x = w - panel_width - 10
        panel_y = 50

        # Draw semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (panel_x, panel_y),
                     (panel_x + panel_width, panel_y + 250),
                     (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

        # Draw title
        cv2.putText(frame, "Probabilities:", (panel_x + 10, panel_y + 25),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

        # Draw bars for each emotion
        for i, (emotion, prob) in enumerate(zip(self.emotions, probabilities)):
            y = panel_y + 50 + i * 30
            bar_length = int(prob * 200)
            color = self.colors[emotion]

            # Draw emotion name
            cv2.putText(frame, f"{emotion[:7]}", (panel_x + 10, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

            # Draw probability bar
            cv2.rectangle(frame, (panel_x + 80, y - 10),
                         (panel_x + 80 + bar_length, y + 5),
                         color, -1)

            # Draw percentage
            cv2.putText(frame, f"{prob*100:.1f}%", (panel_x + 190, y),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)

# ============================================================================
# MAIN
# ============================================================================

if __name__ == "__main__":
    try:
        detector = EmotionDetector()
        detector.run()
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()