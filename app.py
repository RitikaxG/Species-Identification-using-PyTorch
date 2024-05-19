import streamlit as st
from main import ImagePredictor, ImprovedTinyVGG
from PIL import Image
import torch
import torchvision.transforms as transforms

# Try importing generativeai, with alternative comment if unavailable
try:
  import google.generativeai as genai
  from dotenv import load_dotenv  # Assuming dotenv is still used
  import os
except ModuleNotFoundError:
  st.warning("google.generativeai library not found. Text generation disabled.")
  genai = None

# Load environment variables from .env file (if used)
if genai is not None:
  load_dotenv()
  genai.configure(api_key=os.getenv("GOOGLE_API_KEY"))

# Define your custom image transform function
custom_image_transform = transforms.Compose([
    transforms.Resize(size=(64, 64)),
    # Add any other transformations you want to apply
])

# Load your saved model
MODEL_SAVE_PATH = "models/04_pytorch_custom_datasets_model_2.pth"
class_names = ["Abyssinian", "Bengal", "Persian", 'chihuahua', 'pomeranian']
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = ImprovedTinyVGG(input_shape=3, hidden_units=3, output_shape=len(class_names))
model.load_state_dict(torch.load(f=MODEL_SAVE_PATH))
model.to(device)

# Initialize the ImagePredictor
image_predictor = ImagePredictor(model=model, class_names=class_names, device=device)

# Function to generate response using Gemini model (if available)
def get_gemini_response(input):
    if genai is not None:
        try:
            model = genai.GenerativeModel("gemini-pro")
            response = model.generate_content(input)

            # Handle potential JSON parsing errors
            try:
                return response.text  # Convert response to text (assuming it's JSON)
            except ValueError:
                st.error("Error parsing Gemini response. Data might not be JSON.")
                return None

        except Exception as e:
            st.error(f"Error generating response: {e}")
            return None
    else:
        return None  # Indicate text generation unavailable

# Define the Streamlit app
def main():
    st.title("Image Classifier and Allergy Predictor")

    # File upload section
    uploaded_file = st.file_uploader("Upload an image...", type=["jpg", "jpeg", "png"])

    # Predict button
    if uploaded_file is not None:
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_column_width=True)
        if st.button("Classify"):
            # Predict species label
            predicted_label = image_predictor.pred_and_plot_image(uploaded_image=image, custom_image_transform=custom_image_transform)
            st.write(f"Predicted Label: {predicted_label}")

            # Generate prompt for Gemini model
            prompt = f"Predict food allergies for {predicted_label} species."

            # Generate response using Gemini model
            response = get_gemini_response(prompt)
            if response is not None:
                st.subheader("Generated Response:")
                st.write(response)  # Assuming response is text, not a dictionary
            else:
                st.error("An error occurred while generating the response.")

if __name__ == "__main__":
    main()




