#  MindMinder

MindMinder is a Flask-based web application that allows users to record or upload audio meetings and receive intelligent transcripts and summaries. The system performs speaker diarization, transcription (via OpenAI Whisper), and topic-based summarization (via HuggingFace BART model). It also includes email delivery and personal usage statistics.

---
###  Demo
[https://user-images.githubusercontent.com/NAiss1/demoMindminder.mp4](https://github.com/NAiss1/MindMinder/blob/main/demoMindminder.mp4)

---

##  Features

*  **Audio Recording & Uploading**
  Users can record audio with system and mic channels or upload supported audio files.

*  **Speaker Diarization & Transcription**
  Uses Pyannote for diarization and Whisper for transcription.

*  **Smart Summarization**
  Topic segmentation + BART summarization to generate concise, readable summaries.

*  **Email Delivery**
  Sends transcripts and summaries to users via SendGrid.

*  **User Dashboard & Statistics**
  Track usage, recording history, and upload activity with a visual dashboard.

*  **User Authentication**
  Registration, login, profile editing, and password updating.

---



##  Tech Stack

* **Backend**: Python, Flask, PyTorch, HuggingFace Transformers, Pyannote, Whisper, spaCy
* **Frontend**: HTML, CSS, JavaScript (vanilla)
* **Email**: SendGrid API
* **Storage**: JSON file-based database
* **Visualization**: Chart.js for heatmaps and pie charts

---

##  Supported Formats

* Audio: `.wav`, `.mp3`, `.mp4`, `.m4a`, `.ogg`, `.flac`


##  License

MIT License

