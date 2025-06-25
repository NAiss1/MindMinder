import json
import os
import uuid
import datetime
import subprocess
import warnings
import base64
import requests
import torch
import librosa
from tqdm import tqdm
from flask import Flask, Blueprint, request, jsonify, session
from flask_cors import CORS
from flask_bcrypt import Bcrypt
from authentication import auth_bp
from pyannote.audio import Pipeline
import whisper
from transformers import pipeline, AutoModelForSeq2SeqLM, AutoTokenizer

import nltk
nltk.download('punkt')
try:
    nltk.data.find('corpora/stopwords')
except LookupError:
    nltk.download('stopwords')

from sendgrid import SendGridAPIClient
from sendgrid.helpers.mail import Mail, Attachment

import spacy
from sklearn.cluster import KMeans

nlp = spacy.load("en_core_web_md")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATABASE_DIR = os.path.join(BASE_DIR, "database")
os.makedirs(DATABASE_DIR, exist_ok=True)

RECORDINGS_FILE = os.path.join(DATABASE_DIR, "recordings.json")

if not os.path.exists(RECORDINGS_FILE):
    with open(RECORDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump([], f)

def load_recordings():
    with open(RECORDINGS_FILE, "r", encoding="utf-8") as f:
        return json.load(f)

def save_recordings(recordings_list):
    with open(RECORDINGS_FILE, "w", encoding="utf-8") as f:
        json.dump(recordings_list, f, indent=4)

app = Flask(__name__)
app.secret_key = os.urandom(24)
CORS(app)
app.register_blueprint(auth_bp, url_prefix="/auth")

@app.route("/")
def home():
    return "Hello, world!"



SUMMARIZATION_MODEL_NAME = "philschmid/bart-large-cnn-samsum"

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Device set to use: {device}")
if device == "cuda":
    print(f"Using GPU: {torch.cuda.get_device_name(torch.cuda.current_device())}")

print("Loading Whisper model...")
whisper_model = whisper.load_model("base", device=device)

print("Loading Pyannote diarization pipeline...")
diarization_pipeline = Pipeline.from_pretrained(
    "pyannote/speaker-diarization",
    use_auth_token=HUGGINGFACE_TOKEN
)

print("Loading Summarization model (philschmid/bart-large-cnn-samsum)...")
try:
    summarizer_tokenizer = AutoTokenizer.from_pretrained(SUMMARIZATION_MODEL_NAME)
    summarizer_model = AutoModelForSeq2SeqLM.from_pretrained(SUMMARIZATION_MODEL_NAME).to(device)
    summarizer = pipeline(
        "summarization",
        model=summarizer_model,
        tokenizer=summarizer_tokenizer,
        device=0 if device == "cuda" else -1,
        framework="pt"
    )
    print(f"Summarization model '{SUMMARIZATION_MODEL_NAME}' loaded successfully!")
except Exception as e:
    print(f"Error loading summarization model: {e}")
    raise

os.makedirs("uploads", exist_ok=True)

def validate_audio_file(input_path):
    try:
        process = subprocess.run(
            ["ffmpeg", "-i", input_path],
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        output = process.stderr.decode()
        if "Audio" in output:
            print("Valid audio stream detected.")
            return True
        elif "Video" in output:
            print("Invalid audio file: Contains only video stream.")
            return False
        else:
            print("Invalid audio file: No audio or video stream found.")
            return False
    except Exception as e:
        print(f"Error validating audio file: {e}")
        return False

def get_audio_duration(audio_path):
    try:
        duration = librosa.get_duration(path=audio_path) / 3600 
        return round(duration, 2)
    except Exception as e:
        print(f"Error calculating audio duration: {e}")
        return 0

def convert_to_wav(input_path, output_path):
    input_path = os.path.abspath(input_path)
    output_path = os.path.abspath(output_path)
    try:
        process = subprocess.run(
            ["ffmpeg", "-i", input_path, "-vn", "-ar", "16000", "-ac", "1", output_path, "-y"],
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE
        )
        print("FFmpeg Output:", process.stdout.decode())
        return True
    except subprocess.CalledProcessError as e:
        print(f"Error converting file to WAV: {e.stderr.decode()}")
        return False
    except Exception as e:
        print(f"Unexpected error during conversion: {e}")
        return False

def perform_diarization(audio_path):
    print("Performing speaker diarization...")
    diarization = diarization_pipeline(audio_path)
    speaker_segments = []
    for turn, _, speaker in diarization.itertracks(yield_label=True):
        speaker_segments.append({
            "start": turn.start,
            "end": turn.end,
            "speaker": speaker
        })
    print(f"Diarization result: {speaker_segments}")
    return speaker_segments

def transcribe_audio(input_path, segments):
    print("Transcribing audio with Whisper...")
    transcription = ""
    for segment in tqdm(segments, desc="Processing segments", unit="segment"):
        start_time = segment['start']
        end_time = segment['end']
        speaker = segment['speaker']
        segment_audio_path = f"uploads/segment_{start_time}_{end_time}.wav"
        subprocess.run([
            "ffmpeg", "-i", input_path, "-ss", str(start_time), "-to", str(end_time),
            "-c", "copy", segment_audio_path, "-y"
        ])
        try:
            result = whisper_model.transcribe(segment_audio_path, fp16=False, language="en")
            transcription += f"{speaker}: {result['text']}\n"
        except Exception as e:
            print(f"Error transcribing segment {start_time}-{end_time}: {e}")
        if os.path.exists(segment_audio_path):
            os.remove(segment_audio_path)
    return transcription

def convert_to_training_format(transcription_text):
    lines = transcription_text.strip().split('\n')
    structured_data = {"meeting_transcripts": []}
    for line in lines:
        if ': ' in line:
            speaker, content = line.split(': ', 1)
            structured_data["meeting_transcripts"].append({
                "speaker": speaker.strip(),
                "content": content.strip()
            })
    return structured_data


def chunk_text(text, chunk_size):
    words = text.split()
    chunks = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i+chunk_size])
        chunks.append(chunk)
    return chunks

def chunk_summarize(text, chunk_size):

    chunks = chunk_text(text, chunk_size)
    partial_summaries = []
    
    for chunk in chunks:
        if len(chunk.split()) < 30:
            partial_summaries.append(chunk)
            continue

        prompt = f"Summarize this conversation:\n{chunk}"
        summary_chunk = summarizer(
            prompt,
            max_length=100,
            min_length=10,
            num_beams=4,
            repetition_penalty=3.0,
            no_repeat_ngram_size=3,
            early_stopping=True
        )[0]["summary_text"]
        partial_summaries.append(summary_chunk)
    
    final_summary = "\n".join(partial_summaries) if partial_summaries else ""
    
    return {
        "partial_summaries": partial_summaries,
        "final_summary": final_summary
    }

def backup_split_by_length(text, max_words=500):

    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

def segment_into_topics(full_text, n_topics=None, min_topic_length=50):

    doc = nlp(full_text)
    sentences = list(doc.sents)
    if not sentences:
        return [full_text]
    
    sentence_vectors = [sent.vector for sent in sentences]
    
    if n_topics is None:
        n_topics = max(2, int(len(sentences) / 4))
    
    kmeans = KMeans(n_clusters=n_topics, random_state=42)
    clusters = kmeans.fit_predict(sentence_vectors)
    
    topics = []
    current_topic = sentences[0].text
    current_label = clusters[0]
    for i in range(1, len(sentences)):
        if clusters[i] == current_label:
            current_topic += " " + sentences[i].text
        else:
            topics.append(current_topic)
            current_topic = sentences[i].text
            current_label = clusters[i]
    topics.append(current_topic)
    
    merged_topics = []
    for topic in topics:
        if merged_topics and len(topic.split()) < min_topic_length:
            merged_topics[-1] += " " + topic
        else:
            merged_topics.append(topic)
    
    return merged_topics

def summarize_by_topics(structured_transcript):

    full_text = "\n".join(
        f"{turn['speaker']}: {turn['content']}"
        for turn in structured_transcript.get("meeting_transcripts", [])
    )

    print("Segmenting transcript into topics using spaCy...")
    topic_segments = segment_into_topics(full_text)
    print(f"Number of topic segments: {len(topic_segments)}")

    topic_summaries = []
    topic_chunk_summaries = []

    for idx, segment in enumerate(topic_segments):
        word_count = len(segment.split())
        if word_count < 50:
            topic_summaries.append(segment)
            topic_chunk_summaries.append([segment])
            continue

        if word_count > 250:
            subsegments = backup_split_by_length(segment, max_words=250)
            sub_summaries = []
            for sub in subsegments:
                if len(sub.split()) < 30:
                    sub_summaries.append(sub)
                else:
                    prompt = f"Summarize this conversation:\n{sub}"
                    summary_sub = summarizer(
                        prompt,
                        max_length=100,
                        min_length=10,
                        num_beams=4,
                        repetition_penalty=3.0,
                        no_repeat_ngram_size=3,
                        early_stopping=True
                    )[0]["summary_text"]
                    sub_summaries.append(summary_sub)
            final_sum = "\n".join(sub_summaries)
            topic_summaries.append(final_sum)
            topic_chunk_summaries.append(sub_summaries)
        else:
            try:
                chunk_result = chunk_summarize(segment, chunk_size=250)
                partial_sums = chunk_result["partial_summaries"]
                final_sum = chunk_result["final_summary"]

                topic_summaries.append(final_sum)
                topic_chunk_summaries.append(partial_sums)
            except Exception as e:
                print(f"Error summarizing topic segment {idx}: {e}")
                topic_summaries.append("Summary unavailable for this segment.")
                topic_chunk_summaries.append(["Error in chunking."])

    combined_summary = "\n".join(topic_summaries)
    return combined_summary, topic_summaries, topic_chunk_summaries

def summarize_text(jsonl_transcript):
    print("Summarizing structured transcript (fallback)...")
    try:
        dialog_string = "\n".join([
            f"{turn['speaker']}: {turn['content']}"
            for turn in jsonl_transcript["meeting_transcripts"]
        ])
        final_summary = chunk_summarize(dialog_string, chunk_size=500)
        return final_summary
    except Exception as e:
        print(f"Summarization error: {e}")
        return "Summary unavailable due to processing error"

@app.route("/send_recording_email", methods=["POST"])
def send_recording_email():
    user_email = request.args.get("email")
    rec_id = request.args.get("id")
    if not user_email or not rec_id:
        return jsonify({"error": "Missing email or recording ID"}), 400

    all_recordings = load_recordings()
    target_recording = next((r for r in all_recordings if r["user_email"] == user_email and r["id"] == rec_id), None)
    
    if not target_recording:
        return jsonify({"error": "Recording not found"}), 404

    try:
        transcript_text = target_recording["transcript"]
        summary_text = target_recording["summary"]

        email_body = """
        <p>Dear User,</p>
        <p>Please find the <strong>transcript</strong> and <strong>summary</strong> attached.</p>
        <p>Best regards,<br>MindMinder Team</p>
        """

        message = Mail(
            from_email="n.aissauyt@lancaster.ac.uk",
            to_emails=user_email,
            subject=f" Meeting Summary - {rec_id[:8]}",
            html_content=email_body
        )

        transcription_path = os.path.join("uploads", "transcription.txt")
        summary_path = os.path.join("uploads", "summary.txt")

        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(transcript_text)

        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(summary_text)

        with open(transcription_path, "rb") as f:
            transcription_data = base64.b64encode(f.read()).decode()
        with open(summary_path, "rb") as f:
            summary_data = base64.b64encode(f.read()).decode()

        message.add_attachment(Attachment(
            file_content=transcription_data,
            file_type="text/plain",
            file_name="transcription.txt",
            disposition="attachment"
        ))
        message.add_attachment(Attachment(
            file_content=summary_data,
            file_type="text/plain",
            file_name="summary.txt",
            disposition="attachment"
        ))

        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        
        os.remove(transcription_path)
        os.remove(summary_path)

        print(f" Email sent! Status: {response.status_code}")
        return jsonify({"message": "Email sent successfully!"}), 200
    except Exception as e:
        print(" Error sending email:", e)
        return jsonify({"error": str(e)}), 500


def send_email_with_attachments_and_template(recipient, transcription_path, summary_path):
    message = Mail(
        from_email="n.aissauyt@lancaster.ac.uk",
        to_emails=recipient
    )
    message.template_id = SENDGRID_TEMPLATE_ID

    with open(transcription_path, "rb") as f:
        transcription_data = base64.b64encode(f.read()).decode()
    transcription_attachment = Attachment(
        file_content=transcription_data,
        file_type="text/plain",
        file_name="transcription.txt",
        disposition="attachment"
    )
    message.add_attachment(transcription_attachment)

    with open(summary_path, "rb") as f:
        summary_data = base64.b64encode(f.read()).decode()
    summary_attachment = Attachment(
        file_content=summary_data,
        file_type="text/plain",
        file_name="summary.txt",
        disposition="attachment"
    )
    message.add_attachment(summary_attachment)

    try:
        sg = SendGridAPIClient(SENDGRID_API_KEY)
        response = sg.send(message)
        print(f"Email sent successfully! Status code: {response.status_code}")
        return True
    except Exception as e:
        print(f"Error sending email: {e}")
        return False
@app.route("/upload", methods=["POST"])
def upload_audio():
    print("Received Upload Request")
    if "file" not in request.files or "email" not in request.form:
        return jsonify({"error": "File or email not provided"}), 400

    audio_file = request.files["file"]
    user_email = request.form["email"]

    filename_lower = audio_file.filename.lower()
    upload_method = "recording" if filename_lower == "recordingnotupload.wav" else "upload"
    print(f"Processing file: {audio_file.filename} for {user_email}, determined source: {upload_method}")

    original_file_path = os.path.join("uploads", audio_file.filename)
    converted_file_path = os.path.join("uploads", "converted_recording.wav")
    audio_file.save(original_file_path)

    if not validate_audio_file(original_file_path):
        os.remove(original_file_path)
        return jsonify({"error": "Uploaded file does not contain a valid audio stream."}), 400

    if not convert_to_wav(original_file_path, converted_file_path):
        os.remove(original_file_path)
        return jsonify({"error": "File conversion failed."}), 500

    try:
        speaker_segments = perform_diarization(converted_file_path)
        raw_transcription = transcribe_audio(converted_file_path, speaker_segments)
        structured_transcript = convert_to_training_format(raw_transcription)

        final_summary, topic_summaries, _ = summarize_by_topics(structured_transcript)
        combined_summary = "\n".join(topic_summaries)

        transcription_path = os.path.join("uploads", "raw_transcription.txt")
        summary_path = os.path.join("uploads", "summary.txt")

        with open(transcription_path, "w", encoding="utf-8") as f:
            f.write(raw_transcription)
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(combined_summary)

        estimated_hours = get_audio_duration(converted_file_path)
        requests.post("http://127.0.0.1:5000/auth/update_statistics", json={"email": user_email, "hours": estimated_hours})

        os.remove(original_file_path)
        os.remove(converted_file_path)

        recordings = load_recordings()
        new_id = str(uuid.uuid4())
        new_entry = {
            "id": new_id,
            "user_email": user_email,
            "title": request.form.get("title", f"Recording {new_id[:8]}"),
            "timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "transcript": raw_transcription,
            "summary": combined_summary,
            "topic_summaries": topic_summaries,
            "upload_method": upload_method
        }
        recordings.append(new_entry)
        save_recordings(recordings)

        if send_email_with_attachments_and_template(user_email, transcription_path, summary_path):
            print("📧 Email sent successfully after processing!")

        os.remove(transcription_path)
        os.remove(summary_path)

        return jsonify({
            "message": "Processing completed successfully and email sent!",
            "transcript": raw_transcription,
            "summary": combined_summary,
            "topic_summaries": topic_summaries,
            "upload_method": upload_method
        }), 200

    except Exception as e:
        print(f"Error during processing: {e}")
        return jsonify({"error": str(e)}), 500


@app.route("/list_recordings", methods=["GET"])
def list_recordings():
    user_email = request.args.get("email")
    if not user_email:
        return jsonify({"error": "Email is required"}), 400
    all_recordings = load_recordings()
    user_recs = [r for r in all_recordings if r["user_email"] == user_email]
    result = []
    for r in user_recs:
        result.append({
            "id": r["id"],
            "title": r["title"],
            "timestamp": r["timestamp"],
            "upload_method": r.get("upload_method", "upload")
        })
    return jsonify(result), 200

@app.route("/delete_recording", methods=["DELETE"])
def delete_recording():
    user_email = request.args.get("email")
    rec_id = request.args.get("id")
    if not user_email or not rec_id:
        return jsonify({"error": "Missing email or recording id"}), 400
    all_recordings = load_recordings()
    updated_list = []
    deleted = False
    for r in all_recordings:
        if r["user_email"] == user_email and r["id"] == rec_id:
            deleted = True
        else:
            updated_list.append(r)
    if not deleted:
        return jsonify({"error": "Recording not found"}), 404
    save_recordings(updated_list)
    return jsonify({"message": "Recording deleted successfully"}), 200

@app.route("/get_recording", methods=["GET"])
def get_recording():
    user_email = request.args.get("email")
    rec_id = request.args.get("id")
    print(user_email + "\n")
    if not user_email or not rec_id:
        print("ERROR: Missing email or recording id")
        return jsonify({"error": "Missing email or recording id"}), 400
    all_recordings = load_recordings()
    for r in all_recordings:
        if r["user_email"] == user_email and r["id"] == rec_id:
            return jsonify({
                "id": r["id"],
                "title": r["title"],
                "timestamp": r["timestamp"],
                "transcript": r["transcript"],
                "summary": r["summary"],
                "topic_summaries": r.get("topic_summaries", []),
                "upload_method": r.get("upload_method", "upload")
            }), 200
    return jsonify({"error": "Recording not found"}), 404

@app.route("/update_recording_title", methods=["PUT"])
def update_recording_title():
    data = request.json
    email = data.get("email")
    rec_id = data.get("id")
    new_title = data.get("title", "").strip()

    if not email or not rec_id or not new_title:
        return jsonify({"error": "Missing required fields"}), 400

    recordings = load_recordings()
    updated = False
    for r in recordings:
        if r["user_email"] == email and r["id"] == rec_id:
            r["title"] = new_title
            updated = True
            break

    if not updated:
        return jsonify({"error": "Recording not found"}), 404

    save_recordings(recordings)
    return jsonify({"message": "Title updated"}), 200

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)