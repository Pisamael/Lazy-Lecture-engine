from flask import Flask, render_template_string, request, redirect
from werkzeug.utils import secure_filename
import os

app = Flask(__name__)

# Configure your Z: drive folders
UPLOAD_FOLDER = r'Z:\Dev_Workspace\Data\Input\Lecture'
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 100 * 1024 * 1024  # 100MB limit

# Simple Mobile-First UI
HTML_UPLOAD = '''
<!doctype html>
<html lang="en">
<head>
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Lecture Upload</title>
  <style>
    body { font-family: sans-serif; text-align: center; background: white; padding: 20px; }
    .box { border: 2px dashed #FFB6C1; padding: 30px; border-radius: 15px; }
    input { margin: 10px 0; }
    .btn { background: #FFB6C1; color: white; padding: 15px; border: none; border-radius: 8px; width: 100%; }
  </style>
</head>
<body>
  <h1>📤 Upload Lecture</h1>
  <div class="box">
    <form method="post" enctype="multipart/form-data">
      <p>Select Audio or Slides:</p>
      <input type="file" name="files" multiple>
      <input type="submit" class="btn" value="Upload to Legion">
    </form>
  </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def upload_file():
    if request.method == 'POST':
        if 'files' not in request.files:
            return 'No file part'
        
        uploaded_files = request.files.getlist('files')
        for file in uploaded_files:
            if file.filename:
                filename = secure_filename(file.filename)
                file.save(os.path.join(app.config['UPLOAD_FOLDER'], filename))
        
        return '<h1>✅ Files Received on Z: Drive!</h1><a href="/">Upload More</a>'
    
    return render_template_string(HTML_UPLOAD)

if __name__ == '__main__':
    # Runs on port 5000
    app.run(host='0.0.0.0', port=5000)
