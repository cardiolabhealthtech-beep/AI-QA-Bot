import os
import json
import requests
import pandas as pd
from flask import Flask, request, jsonify
from datetime import datetime
from openai import OpenAI
from dotenv import load_dotenv

# -----------------------------------
# 1. Load environment variables
# -----------------------------------
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
CALLYZER_KEY = os.getenv("CALLYZER_SANDBOX_API_KEY")
CALLYZER_BASE = os.getenv("CALLYZER_BASE_URL", "https://sandbox.api.callyzer.co/api/v2.1")

client = OpenAI(api_key=OPENAI_API_KEY)
app = Flask(__name__)

# -----------------------------------
# 2. Helper Functions
# -----------------------------------

def download_recording(url, call_id):
    """Download audio recording from Callyzer or given URL"""
    os.makedirs("recordings", exist_ok=True)
    file_path = f"recordings/{call_id}.mp3"
    print(f"🎧 Downloading: {url}")
    r = requests.get(url, stream=True)
    r.raise_for_status()
    with open(file_path, "wb") as f:
        for chunk in r.iter_content(1024):
            f.write(chunk)
    print("✅ Recording downloaded successfully")
    return file_path


def transcribe_audio(audio_file):
    """Convert speech to text using OpenAI Whisper"""
    print(f"🔊 Transcribing: {audio_file}")
    with open(audio_file, "rb") as f:
        transcript = client.audio.transcriptions.create(model="whisper-1", file=f)
    text = transcript.text
    print("📝 Transcription complete")
    return text


def analyze_call(transcript):
    """Analyze call and give QA score using GPT"""
    print("🤖 Analyzing call...")
    prompt = f"""
    You are a call quality analyst. Analyze this conversation and give:
    - Communication clarity
    - Empathy
    - Product knowledge
    - Professional tone
    Provide a JSON output with 'score' (0–100) and 'feedback'.

    Transcript:
    {transcript}
    """
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "system", "content": "You are an AI QA analyst."},
                  {"role": "user", "content": prompt}]
    )
    result = response.choices[0].message.content
    print("📊 Analysis complete")
    return result


def update_excel(call_id, agent_name, score, feedback, transcript_path):
    """Update or create QA Excel report"""
    os.makedirs("reports", exist_ok=True)
    file_path = "reports/QA_Report.xlsx"

    df_new = pd.DataFrame([{
        "Date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "Call ID": call_id,
        "Agent Name": agent_name,
        "Score": score,
        "Feedback": feedback,
        "Transcript File": transcript_path
    }])

    if os.path.exists(file_path):
        df = pd.read_excel(file_path)
        df = pd.concat([df, df_new], ignore_index=True)
    else:
        df = df_new

    df.to_excel(file_path, index=False)
    print(f"📈 Excel updated → {file_path}")


def callyzer_get(path, params=None):
    """Optional: Pull history from Callyzer Sandbox"""
    url = f"{CALLYZER_BASE}{path}"
    headers = {
        "Authorization": f"Bearer {CALLYZER_KEY}",
        "Content-Type": "application/json"
    }
    r = requests.get(url, headers=headers, params=params or {}, timeout=60)
    r.raise_for_status()
    return r.json()

# -----------------------------------
# 3. Webhook Route (Main Entry)
# -----------------------------------

@app.route("/callyzer-webhook", methods=["POST"])
def handle_callyzer_webhook():
    """Main webhook endpoint for Callyzer"""
    try:
        data = request.get_json(force=True)
        print(f"📩 Webhook received: {data}")

        call_id = data.get("call_id", f"call_{datetime.now().strftime('%H%M%S')}")
        agent_name = data.get("agent_name", "Unknown Agent")
        recording_url = data.get("recording_url")

        if not recording_url:
            return jsonify({"error": "Missing recording_url"}), 400

        # Step 1: Download recording
        audio_file = download_recording(recording_url, call_id)

        # Step 2: Transcribe
        transcript_text = transcribe_audio(audio_file)

        # Step 3: Analyze
        analysis = analyze_call(transcript_text)

        # Step 4: Parse score + feedback from GPT output
        try:
            parsed = json.loads(analysis)
            score = parsed.get("score", "N/A")
            feedback = parsed.get("feedback", "")
        except:
            score = "N/A"
            feedback = analysis

        # Step 5: Save transcript & report
        os.makedirs("transcripts", exist_ok=True)
        transcript_path = f"transcripts/{call_id}.txt"
        with open(transcript_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        os.makedirs("qa_reports", exist_ok=True)
        report_path = f"qa_reports/{call_id}.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump({"score": score, "feedback": feedback}, f, indent=2, ensure_ascii=False)

        # Step 6: Update Excel
        update_excel(call_id, agent_name, score, feedback, transcript_path)

        print(f"✅ QA Report Generated: {call_id}")
        return jsonify({"status": "success", "call_id": call_id, "score": score, "feedback": feedback})

    except Exception as e:
        print("❌ Error:", e)
        return jsonify({"error": str(e)}), 500


# -----------------------------------
# 4. Test Route (to verify online hosting)
# -----------------------------------

@app.route("/", methods=["GET"])
def home():
    return jsonify({"message": "AI-QA Bot Live ✅", "time": datetime.now().isoformat()})


# -----------------------------------
# 5. Run App
# -----------------------------------

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=True)
