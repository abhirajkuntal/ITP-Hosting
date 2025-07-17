from flask import Flask, request, jsonify, send_from_directory, render_template
import os
from run_model import process_image, process_video

app = Flask(__name__)

# Ensure directories exist
os.makedirs('uploads', exist_ok=True)
os.makedirs('processed', exist_ok=True)

@app.route('/')
def homepage():
    return render_template('frontpage.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'})

    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'})

    filename = file.filename
    upload_path = os.path.join('uploads', filename)
    file.save(upload_path)

    print(f"[INFO] Processing file: {upload_path}")

    if filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        output_path = os.path.join('processed', f"output_{filename}")
        process_image(upload_path, output_path)
    elif filename.lower().endswith(('.mp4', '.avi')):
        output_path = os.path.join('processed', f"output_{filename}")
        process_video(upload_path, output_path)
    else:
        return jsonify({'error': 'Unsupported file type'})

    processed_filename = os.path.basename(output_path)
    return jsonify({'result_url': f'/result/{processed_filename}'})

@app.route('/result/<filename>')
def result_page(filename):
    return render_template('resultpage.html', filename=filename)

@app.route('/processed/<filename>')
def processed_file(filename):
    return send_from_directory('processed', filename)

if __name__ == '__main__':
    app.run(debug=True)
