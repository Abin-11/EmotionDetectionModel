import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3
import random
import pickle
import os
import pygame
import mediapipe as mp
import tkinter as tk
from tkinter import messagebox
from tkinter import simpledialog

# Initialize pygame mixer for music
pygame.mixer.init()

# Load the pre-trained model
model = load_model('model_file_30epochs.h5')

# Initialize video capture for webcam
video = cv2.VideoCapture(0)

# Initialize Mediapipe Face Detection
mp_face_detection = mp.solutions.face_detection
mp_drawing = mp.solutions.drawing_utils
face_detection = mp_face_detection.FaceDetection(min_detection_confidence=0.2)

# Dictionary to map emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Initialize the speech engine
engine = pyttsx3.init()

# Conversational responses for each emotion
emotion_responses = {
    'Happy': [
        "You look happy today! What's making you smile?",
        "It's great to see you smiling! What’s the good news?",
        "You're in a good mood today! Keep it up!"
    ],
    'Sad': [
        "I'm sorry to see that you're feeling down. Is there anything you'd like to talk about?",
        "It looks like something is bothering you. Want to chat about it?",
        "I hope things get better for you soon. Do you want to talk?"
    ],
    'Angry': [
        "It seems like something's bothering you. Do you want to share?",
        "You look upset. Would you like to vent about it?",
        "Anger is a strong emotion. Let's talk it out."
    ],
    'Surprise': [
        "You look surprised! What caught you off guard?",
        "Something unexpected happened, didn't it?",
        "That look of surprise is telling! What’s going on?"
    ],
    'Disgust': [
        "Something's clearly not sitting well with you. Want to tell me about it?",
        "You look displeased. What's bothering you?",
        "I see that you're not pleased. What happened?"
    ],
    'Fear': [
        "You seem a bit fearful. Is everything okay?",
        "It looks like something has you worried. Want to talk?",
        "Fear can be tough to handle. Do you want to share your thoughts?"
    ],
    'Neutral': [
        "You seem calm and collected. How are you feeling today?",
        "You look relaxed. How's your day going?",
        "You're looking composed. What's on your mind?"
    ]
}

# Dictionary to map emotions to specific music files
emotion_music = {
    'Happy': 'happy_song.mp3',
    'Sad': 'sad_song.mp3',
    'Angry': 'angry_song.mp3',
    'Surprise': 'surprise_song.mp3',
    'Disgust': 'disgust_song.mp3',
    'Fear': 'fear_song.mp3',
    'Neutral': 'neutral_song.mp3'
}

# Function to play music based on emotion if the user consents
def play_music(emotion):
    music_file = emotion_music.get(emotion, None)
    if music_file:
        # Stop any currently playing music
        pygame.mixer.music.stop()

        # Ask the user if they want to play a song
        user_input = messagebox.askyesno("Play Music", f"Do you want to hear a {emotion.lower()} song?")
        if user_input:
            # Load and play the new music file
            pygame.mixer.music.load(music_file)
            pygame.mixer.music.play()
        else:
            print(f"No song will be played for {emotion}.")

# Function to generate a conversational response based on the detected emotion
def speak_emotion(label):
    response = random.choice(emotion_responses[label])
    engine.say(response)
    engine.runAndWait()

    # Ask and play music based on the emotion detected
    play_music(label)

# Path to store authorized user data
user_data_path = 'authorized_users.pkl'

# Load or initialize the authorized users dictionary
if os.path.exists(user_data_path):
    with open(user_data_path, 'rb') as file:
        authorized_users = pickle.load(file)
else:
    authorized_users = {}

# Function to recognize or register a user
def recognize_or_register_face(face_img):
    face_id = hash(face_img.tobytes())  # Create a unique ID for the face
    if face_id in authorized_users:
        return authorized_users[face_id]  # Return the user's name if recognized
    else:
        # Register the new user
        name = simpledialog.askstring("New User", "New face detected. Please enter your name:")
        if name:
            authorized_users[face_id] = name
            with open(user_data_path, 'wb') as file:
                pickle.dump(authorized_users, file)
            return name
        else:
            return "Unknown"

# Tkinter GUI for login and emotion display
class EmotionApp:
    def __init__(self, root):
        self.root = root
        self.root.title("Facial Emotion Recognition")

        # Login Frame
        self.login_frame = tk.Frame(root)
        self.login_frame.pack(padx=20, pady=20)

        self.name_label = tk.Label(self.login_frame, text="Enter your name:")
        self.name_label.pack()

        self.name_entry = tk.Entry(self.login_frame)
        self.name_entry.pack()

        self.login_button = tk.Button(self.login_frame, text="Login", command=self.login)
        self.login_button.pack()

        # Quit Button
        self.quit_button = tk.Button(self.login_frame, text="Quit", command=self.quit_app)
        self.quit_button.pack()

        # Emotion Display Frame
        self.emotion_frame = tk.Frame(root)
        
        self.emotion_label = tk.Label(self.emotion_frame, text="", font=("Helvetica", 16))
        self.emotion_label.pack()

        self.user_name = ""

    def login(self):
        self.user_name = self.name_entry.get()
        if self.user_name:
            self.login_frame.pack_forget()
            self.emotion_frame.pack(padx=20, pady=20)
            self.start_emotion_detection()
        else:
            messagebox.showwarning("Login", "Please enter your name.")

    def update_emotion_display(self, emotion):
        self.emotion_label.config(text=f"{self.user_name}, Emotion Detected: {emotion}")

    def start_emotion_detection(self):
        while True:
            ret, frame = video.read()
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_detection.process(frame_rgb)

            faces = results.detections
            if faces:
                if len(faces) > 1:
                    print("More than one face detected. Asking new users to register.")
                    for detection in faces:
                        bboxC = detection.location_data.relative_bounding_box
                        h, w, _ = frame.shape
                        x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                        face_img = frame[y:y+height, x:x+width]
                        face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                        resized = cv2.resize(face_img_gray, (48, 48))
                        normalize = resized / 255.0

                        # Check the shape before reshaping
                        if normalize.shape == (48, 48):
                            reshaped = np.reshape(normalize, (1, 48, 48, 1))
                            result = model.predict(reshaped)
                            label = np.argmax(result, axis=1)[0]
                            
                            # Recognize or register the user
                            user_name = recognize_or_register_face(resized)
                            print(f"Hello, {user_name}!")
                            
                            # Get the emotion label and speak it
                            emotion = labels_dict[label]
                            speak_emotion(emotion)

                            # Update the Tkinter interface
                            self.update_emotion_display(emotion)
                            
                            # Draw rectangles around the face and display the emotion label
                            cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 1)
                            cv2.rectangle(frame, (x, y), (x+width, y+height), (50, 50, 255), 2)
                            cv2.rectangle(frame, (x, y-40), (x+width, y), (50, 50, 255), -1)
                            cv2.putText(frame, f"{user_name}: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                elif len(faces) == 1:
                    print("One face detected. Continuing with emotion detection.")
                    detection = faces[0]
                    bboxC = detection.location_data.relative_bounding_box
                    h, w, _ = frame.shape
                    x, y, width, height = int(bboxC.xmin * w), int(bboxC.ymin * h), int(bboxC.width * w), int(bboxC.height * h)
                    face_img = frame[y:y+height, x:x+width]
                    face_img_gray = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY)  # Convert to grayscale
                    resized = cv2.resize(face_img_gray, (48, 48))
                    normalize = resized / 255.0

                    # Check the shape before reshaping
                    if normalize.shape == (48, 48):
                        reshaped = np.reshape(normalize, (1, 48, 48, 1))
                        result = model.predict(reshaped)
                        label = np.argmax(result, axis=1)[0]
                        
                        # Recognize or register the user
                        user_name = recognize_or_register_face(resized)
                        print(f"Hello, {user_name}!")
                        
                        # Get the emotion label and speak it
                        emotion = labels_dict[label]
                        speak_emotion(emotion)

                        # Update the Tkinter interface
                        self.update_emotion_display(emotion)

                        # Draw rectangles around the face and display the emotion label
                        cv2.rectangle(frame, (x, y), (x+width, y+height), (0, 0, 255), 1)
                        cv2.rectangle(frame, (x, y), (x+width, y+height), (50, 50, 255), 2)
                        cv2.rectangle(frame, (x, y-40), (x+width, y), (50, 50, 255), -1)
                        cv2.putText(frame, f"{user_name}: {emotion}", (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            else:
                print("No faces detected.")
    
            # Show the video frame
            cv2.imshow("Frame", frame)
            
            # Break the loop if 'q' is pressed
            k = cv2.waitKey(1)
            if k == ord('q'):
                break

        # Release the video capture and destroy all windows
        video.release()
        cv2.destroyAllWindows()

    def quit_app(self):
        # Method to handle quitting the application
        video.release()
        cv2.destroyAllWindows()
        self.root.quit()

# Initialize Tkinter application
root = tk.Tk()
app = EmotionApp(root)
root.mainloop()
