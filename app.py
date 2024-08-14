import streamlit as st
import requests
import json
import os
import base64
from PIL import Image
import io
import cv2
import numpy as np

API_KEY = os.getenv("GOOGLE_API_KEY")
API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-1.5-pro:generateContent"

def encode_image(image):
    buffered = io.BytesIO()
    image.save(buffered, format="JPEG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def extract_frame(video_file, position):
    try:
        # Reset file pointer to the beginning
        video_file.seek(0)
        
        # Read the entire file content
        file_content = video_file.read()
        
        # Convert to numpy array
        nparr = np.frombuffer(file_content, np.uint8)
        
        # Decode the video file
        cap = cv2.VideoCapture('temp_video.mp4')
        if not cap.isOpened():
            raise Exception("Failed to open video file")
        
        # Set the frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, position)
        
        # Read the frame
        ret, frame = cap.read()
        if not ret:
            raise Exception(f"Failed to read frame at position {position}")
        
        # Convert BGR to RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Convert to PIL Image
        image = Image.fromarray(frame_rgb)
        
        return image
    except Exception as e:
        st.error(f"Failed to extract frame at position {position}. Error: {str(e)}")
        return None

def analyze_form(frames):
    base64_images = [encode_image(frame) for frame in frames if frame is not None]
    
    if not base64_images:
        return {
            "overall_assessment": "Failed to extract any valid frames from the video.",
            "positive_points": [],
            "areas_for_improvement": [],
            "suggestions": ["Please try uploading a different video file."]
        }
    
    headers = {
        "Content-Type": "application/json",
        "x-goog-api-key": API_KEY
    }

    data = {
        "contents": [{
            "parts": [
                {"text": """
                Analyze these images of a person performing a deadlift. The images represent different stages of the lift. Provide detailed feedback on their form throughout the movement, highlighting any issues and suggesting improvements. Focus on key aspects like back position, hip hinge, bar path, and overall body alignment.

                Structure your response as a JSON object with the following keys:
                1. "overall_assessment": A brief overall assessment of the form throughout the lift.
                2. "positive_points": An array of things done correctly.
                3. "areas_for_improvement": An array of areas that need improvement.
                4. "suggestions": An array of specific, actionable suggestions for improving form.

                Ensure your response is valid JSON.
                """},
                *[{"inline_data": {"mime_type": "image/jpeg", "data": img}} for img in base64_images]
            ]
        }],
        "safety_settings": [
            {
                "category": "HARM_CATEGORY_DANGEROUS_CONTENT",
                "threshold": "BLOCK_NONE"
            }
        ],
        "generation_config": {
            "temperature": 0.4,
            "top_p": 1,
            "top_k": 32,
            "max_output_tokens": 2048,
        }
    }

    try:
        response = requests.post(API_URL, headers=headers, json=data)
        response.raise_for_status()
        content = response.json()['candidates'][0]['content']['parts'][0]['text']
        feedback = json.loads(content)
    except requests.RequestException as e:
        st.error(f"API request failed: {str(e)}")
        feedback = {
            "overall_assessment": f"Error: Failed to get response from API. {str(e)}",
            "positive_points": [],
            "areas_for_improvement": [],
            "suggestions": []
        }
    except (KeyError, json.JSONDecodeError) as e:
        st.error(f"Failed to parse API response: {str(e)}")
        feedback = {
            "overall_assessment": "Unable to parse AI response. Please try again.",
            "positive_points": [],
            "areas_for_improvement": [],
            "suggestions": []
        }
    
    return feedback

def main():
    st.title("Deadlift Form Analysis App (Gemini 1.5 Pro)")

    uploaded_file = st.file_uploader("Upload a video of your deadlift", type=["mp4", "mov", "avi"])

    if uploaded_file is not None:
        # Save the uploaded file temporarily
        with open("temp_video.mp4", "wb") as f:
            f.write(uploaded_file.getbuffer())
        
        # Display the video
        st.video("temp_video.mp4")
        
        st.write(f"File size: {uploaded_file.size} bytes")
        st.write(f"File type: {uploaded_file.type}")

        if st.button("Analyze Form"):
            with st.spinner("Analyzing your form..."):
                # Extract frames from the beginning, middle, and end of the video
                cap = cv2.VideoCapture("temp_video.mp4")
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                frames = [
                    extract_frame(uploaded_file, 0),
                    extract_frame(uploaded_file, total_frames // 2),
                    extract_frame(uploaded_file, total_frames - 1)
                ]
                cap.release()
                
                st.write(f"Extracted {sum(1 for f in frames if f is not None)} valid frames")
                
                feedback = analyze_form(frames)
            
            st.subheader("Form Analysis:")
            st.write(feedback["overall_assessment"])
            
            st.subheader("Positive Points:")
            for point in feedback["positive_points"]:
                st.write(f"✅ {point}")
            
            st.subheader("Areas for Improvement:")
            for area in feedback["areas_for_improvement"]:
                st.write(f"🔍 {area}")
            
            st.subheader("Suggestions:")
            for suggestion in feedback["suggestions"]:
                st.write(f"💡 {suggestion}")

    st.warning("Remember: This AI analysis is not a substitute for professional coaching. Always prioritize safety and consult with a qualified trainer for personalized advice.")

# Clean up the temporary file
if os.path.exists("temp_video.mp4"):
    os.remove("temp_video.mp4")

if __name__ == "__main__":
    main()