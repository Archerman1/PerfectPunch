from google import genai
from PIL import Image
import os

client = genai.Client(api_key="API_KEY_HERE")
sys_instruct = """You are a boxing form analysis chatbot. You will receive punch type, 
                                    reaction time of punch and an image of the punch. Based on this input provide the following info in a concise easy to read paragraph:
                                    1. Stance evaluation
                                    2. Punch technique analysis
                                    3. Detailed breakdown of deviations in stance and punch
                                    4. Comparison to professional boxing standards
                                    5. Suggestions for improvement
                                    6. Additional tips for training and practice"""

image_path = os.path.join(os.path.dirname(__file__), "img", "jab.jpg")
image = Image.open(image_path)

punchType = "jab"
reactionTime = "0.5"

userInput = "Analysis for " + punchType + " punch with a reaction time of " + reactionTime + " seconds"

response = client.models.generate_content(
    model="gemini-2.0-flash",
    contents=[image, userInput])

print(response.text)
