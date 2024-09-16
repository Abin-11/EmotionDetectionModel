import cv2
import numpy as np
from tensorflow.keras.models import load_model
import pyttsx3

# Load the pre-trained model
model = load_model('model_file_30epochs.h5')

# Initialize the face detection model
faceDetect = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# Dictionary to map emotion labels
labels_dict = {0: 'Angry', 1: 'Disgust', 2: 'Fear', 3: 'Happy', 4: 'Neutral', 5: 'Sad', 6: 'Surprise'}

# Initialize the speech engine
engine = pyttsx3.init()

# Function to generate a conversational response based on the detected emotion
def speak_emotion(label):
    if label == 'Happy':
        engine.say("You look happy today! What's making you smile?")
    elif label == 'Sad':
        engine.say("I'm sorry to see that you're feeling down. Is there anything you'd like to talk about?")
    elif label == 'Angry':
        engine.say("It seems like something's bothering you. Do you want to share?")
    elif label == 'Surprise':
        engine.say("You look surprised! What caught you off guard?")
    elif label == 'Disgust':
        engine.say("Something's clearly not sitting well with you. Want to tell me about it?")
    elif label == 'Fear':
        engine.say("You seem a bit fearful. Is everything okay?")
    elif label == 'Neutral':
        engine.say("You seem calm and collected. How are you feeling today?")
    engine.runAndWait()

# Load and process the image
frame = cv2.imread("faces5.jpg")
gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
faces = faceDetect.detectMultiScale(gray, 1.3, 3)

# Flag to ensure speech is only triggered once
spoken_once = False

for x, y, w, h in faces:
    if not spoken_once:
        sub_face_img = gray[y:y+h, x:x+w]
        resized = cv2.resize(sub_face_img, (48, 48))
        normalize = resized / 255.0
        reshaped = np.reshape(normalize, (1, 48, 48, 1))
        result = model.predict(reshaped)
        label = np.argmax(result, axis=1)[0]
        
        # Get the emotion label and provide a spoken response
        emotion = labels_dict[label]
        print(emotion)
        speak_emotion(emotion)
        
        # Draw rectangles around the face and display the emotion label
        cv2.rectangle(frame, (x, y), (x+w, y+h), (0, 0, 255), 1)
        cv2.rectangle(frame, (x, y), (x+w, y+h), (50, 50, 255), 2)
        cv2.rectangle(frame, (x, y-40), (x+w, y), (50, 50, 255), -1)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        
        # Mark that we've processed the first face
        spoken_once = True

# Display the image with annotations
cv2.imshow("Frame", frame)
cv2.waitKey(0)
cv2.destroyAllWindows()