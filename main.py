import numpy as np
import pickle
import gradio as gr
import google.generativeai as genai
import os
import folium
import pandas as pd
import matplotlib.pyplot as plt
import requests
from dotenv import load_dotenv

# Load API Keys
load_dotenv()
API_KEY = os.getenv("GEMINI_API_KEY")

if not API_KEY:
    raise ValueError("⚠️ Please set your Gemini API key in a .env file!")

genai.configure(api_key=API_KEY)
model = genai.GenerativeModel("gemini-pro")

# Load trained model & scaler
with open('trained_model.sav', 'rb') as model_file:
    loaded_model = pickle.load(model_file)
with open('scaler.sav', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)


# 🩺 Diabetes Prediction Function
def diabetes_prediction(Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age):
    try:
        input_data = np.array([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age], dtype=float).reshape(1, -1)
        std_data = scaler.transform(input_data)
        prediction = loaded_model.predict(std_data)
        return "🩸 The person is diabetic. Consult a doctor." if prediction[0] == 1 else "✅ The person is not diabetic. Stay healthy!"
    except Exception as e:
        return f"⚠️ Error: {str(e)}"


# 🤖 Chatbot Function
def chatbot_response(user_input, chat_history):
    try:
        response = model.generate_content(user_input)
        chat_history.append(("You", user_input))
        chat_history.append(("DiaChatBoT 🤖", response.text))
        return "", chat_history
    except Exception as e:
        return "", chat_history + [("DiaChatBoT 🤖", f"⚠️ Error: {str(e)}")]


# 📊 Diabetes Risk Graph
def generate_graph():
    data = {"Age": [20, 30, 40, 50, 60], "Diabetes Risk (%)": [10, 20, 40, 60, 80]}
    df = pd.DataFrame(data)

    plt.figure(figsize=(6, 4))
    plt.plot(df["Age"], df["Diabetes Risk (%)"], marker="o", linestyle="-", color="#ff5733", label="Risk Trend")
    plt.fill_between(df["Age"], df["Diabetes Risk (%)"], color="#ffcccb", alpha=0.3)
    plt.xlabel("Age", fontsize=12, color="black")
    plt.ylabel("Diabetes Risk (%)", fontsize=12, color="black")
    plt.title("📈 Diabetes Risk Trends", fontsize=14, color="black", fontweight="bold")
    plt.grid(True, linestyle="--", linewidth=0.5)
    plt.legend()

    # Save image and return the path
    image_path = "risk_trends.png"
    plt.savefig(image_path)
    plt.close()  # Close the plot to free up memory
    return image_path


# 🌟 Build Gradio UI
demo = gr.Blocks(theme="soft")

with demo:
    gr.Markdown("# 🩺 **DiaGuard: AI-Powered Diabetes Prediction & Chatbot**", elem_id="title")
    gr.Markdown("### Made with ❤️ for Humanity <3")

    with gr.Row():
        with gr.Column():
            Pregnancies = gr.Number(label="👶 Pregnancies", value=2, interactive=True)
            Glucose = gr.Number(label="🩸 Glucose Level (mg/dL)", value=120, interactive=True)
            BloodPressure = gr.Number(label="💓 Blood Pressure (mmHg)", value=80, interactive=True)
            SkinThickness = gr.Number(label="📏 Skin Thickness (mm)", value=20, interactive=True)

        with gr.Column():
            Insulin = gr.Number(label="💉 Insulin Level (IU/mL)", value=85, interactive=True)
            BMI = gr.Number(label="⚖️ BMI (kg/m²)", value=25.6, interactive=True)
            DiabetesPedigreeFunction = gr.Number(label="🧬 Diabetes Pedigree Function", value=0.5, interactive=True)
            Age = gr.Number(label="🎂 Age", value=35, interactive=True)

    output = gr.Textbox(label="🩺 Prediction Result", interactive=False)
    predict_btn = gr.Button("🔍 Predict", variant="primary")
    predict_btn.click(diabetes_prediction, inputs=[Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DiabetesPedigreeFunction, Age], outputs=output)

    gr.Markdown("---")

    # 🤖 Chatbot UI with Background Image
    chatbot_ui = gr.Chatbot(label="💬 Chat with DiaChatBoT", height=400)
    user_input = gr.Textbox(label="👨‍⚕️ Ask about diabetes:")
    chat_btn = gr.Button("💬 Send", variant="primary")

    user_input.submit(chatbot_response, inputs=[user_input, chatbot_ui], outputs=[user_input, chatbot_ui])
    chat_btn.click(chatbot_response, inputs=[user_input, chatbot_ui], outputs=[user_input, chatbot_ui])

    gr.Markdown("---")

    # 📊 Interactive Graph
    gr.Markdown("## 📊 **Diabetes Risk Trends**")
    graph_btn = gr.Button("📊 Show Graph", variant="secondary")
    graph_output = gr.Image(label="📈 Diabetes Risk Trends")
    graph_btn.click(generate_graph, inputs=[], outputs=graph_output)

    gr.Markdown("---")


# 🚀 Launch App
if __name__ == "__main__":
    demo.launch()
