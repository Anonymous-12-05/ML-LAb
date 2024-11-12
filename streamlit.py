import streamlit as st
import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load the custom pose estimation model
model = load_model("pose_estimation.h5")

st.title("Real-time Pose Estimation using Custom Model")

# Display instructions
st.write("Click on 'Start Camera' to begin pose detection.")

# Preprocess function for model input
def preprocess_frame(frame, input_shape):
    # Resize frame to model's expected input size
    resized_frame = cv2.resize(frame, (input_shape[1], input_shape[0]))
    # Normalize the frame (scale pixel values if needed by your model)
    normalized_frame = resized_frame / 255.0
    # Add batch dimension
    return np.expand_dims(normalized_frame, axis=0)

# Function to apply model on frame and get keypoints
def get_keypoints(frame):
    input_shape = model.input_shape[1:3]  # Model input size (height, width)
    preprocessed_frame = preprocess_frame(frame, input_shape)
    keypoints = model.predict(preprocessed_frame)[0]  # Predict keypoints
    return keypoints.reshape(-1, 2)  # Reshape to (num_keypoints, 2)

# Draw keypoints on frame
def draw_keypoints(frame, keypoints):
    for (x, y) in keypoints:
        # Scale coordinates back to original frame size
        h, w, _ = frame.shape
        x, y = int(x * w), int(y * h)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)
    return frame

# Set up a button to start/stop the camera feed
if st.button("Start Camera"):
    cap = cv2.VideoCapture(0)
    
    if not cap.isOpened():
        st.write("Error: Could not open camera.")
    else:
        st.write("Press 'Stop Camera' to end the video capture.")

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            st.write("Failed to capture image")
            break
        
        # Get keypoints from model prediction
        keypoints = get_keypoints(frame)
        
        # Draw keypoints on the frame
        frame_with_keypoints = draw_keypoints(frame, keypoints)
        
        # Display the frame with keypoints
        st.image(cv2.cvtColor(frame_with_keypoints, cv2.COLOR_BGR2RGB), channels="RGB")

        # Optionally, display keypoints data
        st.write("Detected Keypoints:", keypoints)

        # Stop capturing when 'Stop Camera' is pressed
        if st.button("Stop Camera"):
            break

    # Release the camera
    cap.release()
    st.write("Camera stopped.")
