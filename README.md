
Facial Emotion Detection System with User Authentication and Music Recommendation
This project is a comprehensive Facial Emotion Detection System using TensorFlow, Keras, OpenCV, and Mediapipe. It detects facial emotions from live video input or images and provides conversational responses with text-to-speech, user authentication, and emotion-based music recommendation. The application is integrated with a Tkinter GUI for user login and emotion display.

Features
Detects seven emotions: Angry, Disgust, Fear, Happy, Neutral, Sad, Surprise.
User authentication via face recognition.
Plays emotion-specific music upon user consent.
Text-to-speech conversational responses based on the detected emotion.
Tkinter-based user interface for login and emotion display.
Requirements
Python 3.6+
TensorFlow 2.x
Keras
OpenCV
Pyttsx3
Mediapipe
Tkinter
Pygame
Installation
Clone the repository:

bash
Copy code
git clone https://github.com/your-repo/facial-emotion-detection.git
cd facial-emotion-detection
Install the required packages:

bash
Copy code
pip install -r requirements.txt
Download and place the haarcascade_frontalface_default.xml file for face detection in the root directory:

bash
Copy code
wget https://github.com/opencv/opencv/blob/master/data/haarcascades/haarcascade_frontalface_default.xml
Prepare your music files for each emotion and place them in the project folder. The file names should match the emotions:

happy_song.mp3
sad_song.mp3
angry_song.mp3
surprise_song.mp3
disgust_song.mp3
fear_song.mp3
neutral_song.mp3
Training the Emotion Detection Model
Set the directory for your training and testing datasets:

python
Copy code
train_data_dir='Facial Emotion Recognition/train/'
validation_data_dir='Facial Emotion Recognition/test/'
The model architecture is based on Convolutional Neural Networks (CNNs) with multiple Conv2D, MaxPooling2D, and Dropout layers. It is trained on grayscale images of size (48, 48) from the dataset.

Run the training script to train the model:

bash
Copy code
python train_model.py
The trained model is saved as model_file.h5.

Running the Application
Start the application by running the main.py script:

bash
Copy code
python main.py
The Tkinter GUI will prompt the user to log in with their name.

The system will capture real-time video input from your webcam. If one face is detected, the system will perform emotion detection and provide conversational responses. If more than one face is detected, it will prompt the new user for authentication.

The detected emotion will trigger a corresponding music recommendation, asking the user whether they would like to play a song based on their emotional state.

Face Recognition and User Registration
The application will check if a detected face is already registered in the system using a hash of the face image.
If the face is unregistered, it will prompt the user to enter their name and save the face for future recognition.
Conversational Responses
Each detected emotion triggers a unique conversational response via text-to-speech (using pyttsx3). The system responds with random pre-defined responses for each emotion.

Examples:

Happy: "You look happy today! What's making you smile?"
Sad: "I'm sorry to see that you're feeling down. Is there anything you'd like to talk about?"
Emotion-Based Music Recommendation
After detecting an emotion, the system will ask the user if they would like to listen to a song based on their emotional state. If the user agrees, the corresponding emotion-specific song will be played using Pygame.

File Structure
bash
Copy code
facial-emotion-detection/
│
├── train_model.py            # Training script for the model
├── main.py                   # Main application script
├── model_file.h5             # Pre-trained model (after training)
├── requirements.txt          # Required packages
├── happy_song.mp3            # Music files for emotion-specific responses
├── sad_song.mp3
├── angry_song.mp3
├── surprise_song.mp3
├── disgust_song.mp3
├── fear_song.mp3
├── neutral_song.mp3
├── haarcascade_frontalface_default.xml # Haarcascade file for face detection
└── authorized_users.pkl      # Pickle file to store authorized users
Future Enhancements
Expand the emotion classification model for better accuracy.
Integrate additional features like multi-user tracking.
Add support for more emotions and corresponding actions.
Improve the Tkinter GUI for a more seamless user experience.
Acknowledgments
TensorFlow/Keras: For providing the model architecture.
OpenCV: For face detection.
Pyttsx3: For enabling text-to-speech functionality.
Pygame: For handling audio playback.
