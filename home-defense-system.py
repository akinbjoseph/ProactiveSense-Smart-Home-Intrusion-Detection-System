# Proactive Home Defense System
# Description: A home security system using facial recognition and motion tracking with Raspberry Pi

import os
import cv2
import numpy as np
import smtplib
import time
import logging
import threading
from email.mime.multipart import MIMEMultipart
from email.mime.text import MIMEText
from email.mime.image import MIMEImage
import face_recognition
import pickle
from gpiozero import Servo
from time import sleep
import RPi.GPIO as GPIO

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("home_defense.log"),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("HomeDefenseSystem")

class FaceRecognizer:
    """
    Handles face detection and recognition using face_recognition library
    """
    def __init__(self, database_path="faces_db", confidence_threshold=0.6):
        """
        Initialize the face recognizer
        
        Args:
            database_path (str): Path to store/load face encodings
            confidence_threshold (float): Threshold for face recognition confidence
        """
        self.known_face_encodings = []
        self.known_face_names = []
        self.database_path = database_path
        self.confidence_threshold = confidence_threshold
        
        # Create database directory if it doesn't exist
        if not os.path.exists(database_path):
            os.makedirs(database_path)
            logger.info(f"Created face database directory at {database_path}")
        
        # Load existing face database if available
        self.load_face_database()
        
    def load_face_database(self):
        """
        Load known faces from the database
        """
        encoding_file = os.path.join(self.database_path, "encodings.pkl")
        if os.path.exists(encoding_file):
            try:
                with open(encoding_file, 'rb') as f:
                    data = pickle.load(f)
                    self.known_face_encodings = data["encodings"]
                    self.known_face_names = data["names"]
                logger.info(f"Loaded {len(self.known_face_names)} faces from database")
            except Exception as e:
                logger.error(f"Error loading face database: {e}")
        else:
            logger.info("No existing face database found")
    
    def save_face_database(self):
        """
        Save current face database to disk
        """
        encoding_file = os.path.join(self.database_path, "encodings.pkl")
        try:
            with open(encoding_file, 'wb') as f:
                data = {"encodings": self.known_face_encodings, "names": self.known_face_names}
                pickle.dump(data, f)
            logger.info(f"Saved {len(self.known_face_names)} faces to database")
        except Exception as e:
            logger.error(f"Error saving face database: {e}")
    
    def add_face(self, image, name):
        """
        Add a new face to the database
        
        Args:
            image (numpy.ndarray): Image containing the face
            name (str): Name of the person
        
        Returns:
            bool: True if face was added successfully, False otherwise
        """
        try:
            # Find face locations in the image
            face_locations = face_recognition.face_locations(image)
            if not face_locations:
                logger.warning("No face found in the provided image")
                return False
            
            # Get face encoding for the first face found
            face_encoding = face_recognition.face_encodings(image, face_locations)[0]
            
            # Add to our lists
            self.known_face_encodings.append(face_encoding)
            self.known_face_names.append(name)
            
            # Save updated database
            self.save_face_database()
            logger.info(f"Added face for {name} to database")
            return True
            
        except Exception as e:
            logger.error(f"Error adding face: {e}")
            return False
    
    def recognize_faces(self, frame):
        """
        Recognize faces in the given frame
        
        Args:
            frame (numpy.ndarray): Image frame to process
            
        Returns:
            list: List of (name, location) tuples for each face detected
        """
        # Resize frame for faster processing (0.25x original size)
        small_frame = cv2.resize(frame, (0, 0), fx=0.25, fy=0.25)
        
        # Convert the image from BGR color (OpenCV) to RGB (face_recognition)
        rgb_small_frame = cv2.cvtColor(small_frame, cv2.COLOR_BGR2RGB)
        
        # Find all faces in the current frame
        face_locations = face_recognition.face_locations(rgb_small_frame)
        face_encodings = face_recognition.face_encodings(rgb_small_frame, face_locations)
        
        face_results = []
        
        for face_encoding, face_location in zip(face_encodings, face_locations):
            # Scale back up face locations since we scaled the image down
            top, right, bottom, left = [coord * 4 for coord in face_location]
            
            # See if the face matches any known faces
            if not self.known_face_encodings:
                name = "Unknown"
            else:
                # Compare with known faces
                face_distances = face_recognition.face_distance(self.known_face_encodings, face_encoding)
                best_match_idx = np.argmin(face_distances)
                
                if face_distances[best_match_idx] < self.confidence_threshold:
                    name = self.known_face_names[best_match_idx]
                else:
                    name = "Unknown"
            
            face_results.append((name, (top, right, bottom, left)))
            
        return face_results
        

class TurretController:
    """
    Controls the physical turret with pan and tilt servos
    """
    def __init__(self, pan_pin=12, tilt_pin=13, laser_pin=25):
        """
        Initialize the turret controller
        
        Args:
            pan_pin (int): GPIO pin for pan servo
            tilt_pin (int): GPIO pin for tilt servo
            laser_pin (int): GPIO pin for laser
        """
        # Setup GPIO
        GPIO.setmode(GPIO.BCM)
        GPIO.setwarnings(False)
        
        # Setup laser pin
        self.laser_pin = laser_pin
        GPIO.setup(self.laser_pin, GPIO.OUT)
        GPIO.output(self.laser_pin, GPIO.LOW)
        
        # Setup servos
        self.pan_servo = Servo(pan_pin)
        self.tilt_servo = Servo(tilt_pin)
        
        # Center the turret
        self.center()
        
        logger.info("Turret controller initialized")
    
    def center(self):
        """
        Center the turret
        """
        self.pan_servo.value = 0
        self.tilt_servo.value = 0
        logger.info("Turret centered")
        
    def point_at(self, x_ratio, y_ratio):
        """
        Point the turret at a specific position
        
        Args:
            x_ratio (float): Horizontal position (0.0 to 1.0, left to right)
            y_ratio (float): Vertical position (0.0 to 1.0, top to bottom)
        """
        # Convert ratios to servo values (-1 to 1)
        # Adjust the conversion based on your camera field of view and turret mechanics
        pan_value = (x_ratio * 2) - 1
        # Invert y-axis (camera y increases downward)
        tilt_value = 1 - (y_ratio * 2)
        
        # Limit values to safe range
        pan_value = max(-1, min(1, pan_value))
        tilt_value = max(-1, min(1, tilt_value))
        
        # Set servo positions
        self.pan_servo.value = pan_value
        self.tilt_servo.value = tilt_value
        
    def activate_laser(self, activate=True):
        """
        Activate or deactivate the laser
        
        Args:
            activate (bool): True to turn on, False to turn off
        """
        GPIO.output(self.laser_pin, GPIO.HIGH if activate else GPIO.LOW)
        logger.info(f"Laser {'activated' if activate else 'deactivated'}")
    
    def cleanup(self):
        """
        Clean up GPIO resources
        """
        self.activate_laser(False)
        GPIO.cleanup()
        logger.info("Turret controller resources cleaned up")


class NotificationSystem:
    """
    Handles sending notifications when threats are detected
    """
    def __init__(self, sender_email, sender_password, recipient_email):
        """
        Initialize notification system
        
        Args:
            sender_email (str): Email address to send alerts from
            sender_password (str): Password for sender email
            recipient_email (str): Email address to send alerts to
        """
        self.sender_email = sender_email
        self.sender_password = sender_password
        self.recipient_email = recipient_email
        logger.info("Notification system initialized")
    
    def send_alert(self, message, image=None):
        """
        Send an email alert with optional image attachment
        
        Args:
            message (str): Alert message
            image (numpy.ndarray, optional): Image to attach
        
        Returns:
            bool: True if alert was sent successfully, False otherwise
        """
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.sender_email
            msg['To'] = self.recipient_email
            msg['Subject'] = "HOME SECURITY ALERT"
            
            # Add message body
            msg.attach(MIMEText(message, 'plain'))
            
            # Add image if provided
            if image is not None:
                # Convert OpenCV image to bytes
                _, img_encoded = cv2.imencode('.jpg', image)
                img_bytes = img_encoded.tobytes()
                
                # Attach image
                image_attachment = MIMEImage(img_bytes)
                image_attachment.add_header('Content-Disposition', 'attachment', filename='intruder.jpg')
                msg.attach(image_attachment)
            
            # Connect to SMTP server and send email
            with smtplib.SMTP('smtp.gmail.com', 587) as server:
                server.starttls()
                server.login(self.sender_email, self.sender_password)
                server.send_message(msg)
            
            logger.info(f"Alert email sent to {self.recipient_email}")
            return True
            
        except Exception as e:
            logger.error(f"Error sending alert: {e}")
            return False


class HomeDefenseSystem:
    """
    Main class that integrates face recognition, turret control, and notifications
    """
    def __init__(self, camera_source=0, email_config=None):
        """
        Initialize the home defense system
        
        Args:
            camera_source (int or str): Camera index or IP camera URL
            email_config (dict): Email configuration for notifications
        """
        # Initialize camera
        self.camera = cv2.VideoCapture(camera_source)
        if not self.camera.isOpened():
            raise ValueError(f"Could not open camera source: {camera_source}")
        
        # Get camera resolution
        self.frame_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
        logger.info(f"Camera initialized with resolution {self.frame_width}x{self.frame_height}")
        
        # Initialize components
        self.face_recognizer = FaceRecognizer()
        self.turret = TurretController()
        
        # Initialize notification system if email config provided
        self.notification = None
        if email_config:
            self.notification = NotificationSystem(
                email_config.get('sender_email', ''),
                email_config.get('sender_password', ''),
                email_config.get('recipient_email', '')
            )
        
        # System state
        self.running = False
        self.current_threats = {}  # Dictionary to track current threats
        self.last_notification_time = 0
        self.notification_cooldown = 60  # Seconds between notifications
        
        logger.info("Home Defense System initialized")
    
    def add_known_person(self):
        """
        Capture and add a known person to the face database
        
        Returns:
            bool: True if person was added successfully, False otherwise
        """
        name = input("Enter the person's name: ")
        logger.info(f"Adding {name} to known faces...")
        
        # Capture multiple samples for better recognition
        faces_captured = 0
        while faces_captured < 5:
            print(f"Capturing sample {faces_captured+1}/5. Position face in camera.")
            input("Press Enter when ready...")
            
            # Capture frame
            ret, frame = self.camera.read()
            if not ret:
                logger.error("Failed to capture image")
                return False
            
            # Display the frame
            cv2.imshow("Capture Face", frame)
            cv2.waitKey(1)
            
            # Add face to recognizer
            if self.face_recognizer.add_face(frame, name):
                faces_captured += 1
                print(f"Sample {faces_captured} captured!")
            else:
                print("No face detected. Please try again.")
            
            time.sleep(1)
        
        cv2.destroyAllWindows()
        logger.info(f"Successfully added {name} to known faces")
        return True
    
    def process_frame(self, frame):
        """
        Process a video frame for face recognition and threat detection
        
        Args:
            frame (numpy.ndarray): Video frame to process
            
        Returns:
            numpy.ndarray: Processed frame with annotations
        """
        # Make a copy for drawing
        display_frame = frame.copy()
        
        # Recognize faces
        face_results = self.face_recognizer.recognize_faces(frame)
        
        # Track current threats
        current_threats_this_frame = {}
        
        for name, (top, right, bottom, left) in face_results:
            # Draw rectangle around face
            color = (0, 255, 0) if name != "Unknown" else (0, 0, 255)  # Green for known, Red for unknown
            cv2.rectangle(display_frame, (left, top), (right, bottom), color, 2)
            
            # Draw name
            cv2.putText(display_frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # If person is unknown, consider them a threat
            if name == "Unknown":
                # Calculate position in frame
                face_center_x = (left + right) / 2
                face_center_y = (top + bottom) / 2
                
                # Convert to ratios (0-1)
                x_ratio = face_center_x / self.frame_width
                y_ratio = face_center_y / self.frame_height
                
                # Add to current threats
                threat_id = f"{left}_{top}"  # Simple unique identifier
                current_threats_this_frame[threat_id] = {
                    "position": (x_ratio, y_ratio),
                    "time_detected": time.time(),
                    "frame_position": (top, right, bottom, left)
                }
        
        # Update the current threats dictionary
        self.current_threats = current_threats_this_frame
        
        # Handle threats
        self._handle_threats(display_frame)
        
        return display_frame
    
    def _handle_threats(self, frame):
        """
        Handle detected threats by controlling turret and sending notifications
        
        Args:
            frame (numpy.ndarray): Current video frame
        """
        if not self.current_threats:
            # No threats, deactivate laser and center turret
            self.turret.activate_laser(False)
            self.turret.center()
            return
        
        # Get the first threat (in a more sophisticated system, you might prioritize threats)
        threat_id, threat_data = next(iter(self.current_threats.items()))
        
        # Point turret at the threat
        x_ratio, y_ratio = threat_data["position"]
        self.turret.point_at(x_ratio, y_ratio)
        
        # Activate laser
        self.turret.activate_laser(True)
        
        # Draw targeting indicator on the frame
        h, w = frame.shape[:2]
        target_x = int(x_ratio * w)
        target_y = int(y_ratio * h)
        cv2.circle(frame, (target_x, target_y), 20, (0, 0, 255), 2)
        cv2.line(frame, (target_x - 30, target_y), (target_x + 30, target_y), (0, 0, 255), 2)
        cv2.line(frame, (target_x, target_y - 30), (target_x, target_y + 30), (0, 0, 255), 2)
        
        # Send notification if cooldown has elapsed
        current_time = time.time()
        if (self.notification and 
            current_time - self.last_notification_time > self.notification_cooldown):
            
            # Extract threat region
            top, right, bottom, left = threat_data["frame_position"]
            # Add some padding
            padding = 50
            top = max(0, top - padding)
            left = max(0, left - padding)
            bottom = min(frame.shape[0], bottom + padding)
            right = min(frame.shape[1], right + padding)
            
            threat_image = frame[top:bottom, left:right]
            
            # Send notification
            message = (
                "SECURITY ALERT: Unknown person detected!\n\n"
                f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}\n"
                "The defense system is tracking the intruder."
            )
            
            # Send in a separate thread to avoid blocking
            threading.Thread(
                target=self.notification.send_alert,
                args=(message, threat_image)
            ).start()
            
            self.last_notification_time = current_time
    
    def run(self):
        """
        Run the home defense system in a loop
        """
        self.running = True
        logger.info("Home Defense System is running")
        
        try:
            while self.running:
                # Read frame
                ret, frame = self.camera.read()
                if not ret:
                    logger.error("Failed to grab frame")
                    break
                
                # Process frame
                processed_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow("Home Defense System", processed_frame)
                
                # Check for key press
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('a'):
                    self.add_known_person()
                    
        finally:
            self.cleanup()
    
    def cleanup(self):
        """
        Clean up resources
        """
        self.running = False
        self.camera.release()
        cv2.destroyAllWindows()
        self.turret.cleanup()
        logger.info("Home Defense System shutdown complete")


def main():
    # Email configuration (replace with your own credentials)
    email_config = {
        'sender_email': 'your_email@gmail.com',
        'sender_password': 'your_app_password',  # Use app password for Gmail
        'recipient_email': 'your_email@gmail.com'
    }
    
    # Camera source (0 for webcam, or IP camera URL)
    camera_source = 0
    
    try:
        # Initialize and run the system
        system = HomeDefenseSystem(camera_source, email_config)
        
        print("Home Defense System")
        print("-------------------")
        print("Press 'a' to add a known person")
        print("Press 'q' to quit")
        
        # Run the main loop
        system.run()
        
    except KeyboardInterrupt:
        print("Shutting down...")
    except Exception as e:
        logger.error(f"Error: {e}")
        
if __name__ == "__main__":
    main()
