import os
import json
import requests
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# ==================== LOAD ENVIRONMENT VARIABLES ====================
load_dotenv()  # Load .env file

# ==================== CONFIGURATION ====================
app = Flask(__name__)

# Load API key from .env
openai_api_key = os.getenv("OPENAI_API_KEY")
if not openai_api_key:
    raise Exception("❌ OPENAI_API_KEY not found in .env file!")

client = OpenAI(api_key=openai_api_key)

# Create required folders
os.makedirs("recordings", exist_ok=True)
os.makedirs("transcripts", exist_ok=True)
os.makedirs("qa_reports", exist_ok=True)

# ==================== HELPER FUNCTIONS ====================

def download_recording(recording_url, call_id):
    """Download call recording"""
    filename = f"recordings/{call_id}.mp3"
    print(f"🎧 Downloading: {recording_url}")
    response = requests.get(recording_url)
    if response.status_code == 200:
        with open(filename, "wb") as f:
            f.write(response.content)
        print("✅ Recording downloaded successfully")
        return filename
    else:
        raise Exception(f"Failed to download recording: {response.status_code}")

def transcribe_audio(audio_file):
    """Transcribe using Whisper"""
    print(f"🔊 Transcribing: {audio_file}")
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(
            model="whisper-1",
            file=f
        )
    print("📝 Transcription complete")
    return transcript.text

def analyze_call(transcript):
    """Analyze call quality using GPT"""
    print("🤖 Analyzing call...")
    prompt = f"""
You are a call quality evaluation assistant.
Analyze the following call transcript and give a detailed score (0-100) with feedback.

Transcript:
{transcript}

Return response in pure JSON like this:
{{
  "score": <number>,
  "feedback": "<short feedback>"
}}
"""
    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": "You are a helpful QA evaluation assistant."},
            {"role": "user", "content": prompt}
        ]
    )
    result = completion.choices[0].message.content
    print("📊 Analysis complete")
    return result

def update_excel(call_id, agent_name, score, feedback, transcript_path):
    """Add or update Excel report"""
    excel_file = "QA_Report.xlsx"

    # If Excel doesn't exist, create new
    if not os.path.exists(excel_file):
        df = pd.DataFrame(columns=["Call ID", "Agent Name", "Date", "Score", "Feedback", "Transcript File"])
    else:
        df = pd.read_excel(excel_file)

    # New row data
    new_entry = {
        "Call ID": call_id,
        "Agent Name": agent_name,
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Score": score,
        "Feedback": feedback,
        "Transcript File": os.path.abspath(transcript_path)
    }

    # Add row and save
    df = pd.concat([df, pd.DataFrame([new_entry])], ignore_index=True)
    df.to_excel(excel_file, index=False)
    print(f"📈 Excel updated → {excel_file}")

# ==================== WEBHOOK ENDPOINT ====================

@app.route("/callyzer-webhook", methods=["POST"])
def handle_callyzer_webhook():
    try:
        data = request.get_json()
        print("📩 Webhook received:", data)

        call_id = data.get("call_id")
        recording_url = data.get("recording_url")
        agent_name = data.get("agent_name", "Unknown")

        if not recording_url or not call_id:
            return jsonify({"error": "Missing call_id or recording_url"}), 400

        # Step 1: Download recording
        audio_file = download_recording(recording_url, call_id)

        # Step 2: Transcribe
        transcript_text = transcribe_audio(audio_file)

        # Step 3: Analyze QA
        analysis = analyze_call(transcript_text)

        # Step 4: Extract score and feedback safely
        try:
            result = json.loads(analysis)
            score = result.get("score", "N/A")
            feedback = result.get("feedback", "")
        except:
            score = "N/A"
            feedback = analysis

        # Step 5: Save transcript and report
        transcript_path = f"transcripts/{call_id}.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        with open(f"qa_reports/{call_id}.json", "w", encoding="utf-8") as f:
            f.write(analysis)

        # Step 6: Update Excel
        update_excel(call_id, agent_name, score, feedback, transcript_path)

        print(f"✅ QA Report Generated for: {call_id}")
        return jsonify({"status": "success", "score": score, "feedback": feedback})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500

@app.route("/", methods=["GET"])
def home():
    return "🚀 Callyzer AI QA Bot (Excel Reporting + .env Integration) ✅"

# ==================== RUN SERVER ====================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=True)
