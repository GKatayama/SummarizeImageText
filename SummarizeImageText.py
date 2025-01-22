import os
import io
import time
import json
import streamlit
from azure.cognitiveservices.vision.computervision import ComputerVisionClient
from msrest.authentication import CognitiveServicesCredentials
from openai import AzureOpenAI
from dotenv import load_dotenv
from PIL import Image

load_dotenv()

azure_computer_vision_api_key = os.getenv("AZURE_COMPUTER_VISION_API_KEY")
openai_api_key = os.getenv("AZURE_OPENAI_API_KEY")

def generate_summary(context):
    client = AzureOpenAI(
        api_version="2024-10-21",
        azure_endpoint="https://excel-augmented-generation-openai.openai.azure.com/openai/deployments/gpt-4/chat/completions?api-version=2024-08-01-preview"
    )
    prompt = f"Look at the following text: \n{context}\n Please summarize the text in as few sentences as possible without losing the main idea. \n\nSummary: "
    completion = client.chat.completions.create(
        model="gpt-4",
        messages=[
            {
                "role": "user",
                "content": prompt,
            },
        ],
    )
    return completion.to_json()

def ocr():
    azure_computer_vision_endpoint = "https://ocr-garret-computervision.cognitiveservices.azure.com/"
    client = ComputerVisionClient(azure_computer_vision_endpoint, CognitiveServicesCredentials(azure_computer_vision_api_key))

    streamlit.title("OCR with Azure Computer Vision")
    image = streamlit.file_uploader("Upload an image", type=["jpg", "jpeg", "png"])

    if image is not None:
        image = Image.open(image)
        streamlit.image(image, caption="Uploaded Image", use_container_width=True)
        streamlit.write("")
        streamlit.write("Generating summary...")

        image_bytes = io.BytesIO()
        image.save(image_bytes, format=image.format)
        image_bytes = image_bytes.getvalue()

        response = client.read_in_stream(io.BytesIO(image_bytes), raw=True)
        operation_location_remote = response.headers["Operation-Location"]
        operation_id = operation_location_remote.split("/")[-1]

        while True:
            results = client.get_read_result(operation_id)
            if results.status not in ['notStarted', 'running']:
                break
            time.sleep(1)

        if results.status == "succeeded":
            context = ""
            for text_result in results.analyze_result.read_results:
                for line in text_result.lines:
                    context += line.text + "\n"
            return context
        else:
            streamlit.write("Failed with status: {}".format(results.status))
            return None

if __name__ == "__main__":
    context = ocr()
    if context is not None:
        summary = json.loads(generate_summary(context))
        streamlit.write(summary["choices"][0]["message"]["content"])