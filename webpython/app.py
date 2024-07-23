from flask import Flask, send_from_directory, render_template

app = Flask(__name__)

# HTTP_PORT = int(os.getenv("HTTP_PORT", 8000))

# Serve the audio client HTML file
@app.route('/audio')
def serve_audio():
    return render_template('audio_client.html')

# Run the server
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)
