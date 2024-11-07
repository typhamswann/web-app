# app.py
import os
import shutil
import zipfile
from flask import Flask, render_template, request, send_file, redirect, Response, stream_with_context
from pathlib import Path
import uuid
from run_pipeline import run_full_pipeline_generator  # Import the generator function
import threading
import time

app = Flask(__name__)

# Inside your process route
job_id = str(uuid.uuid4())

UPLOAD_FOLDER = f'uploads/{job_id}'
PROCESSED_FOLDER = f'processed/{job_id}'
ZIP_FOLDER = f'zipped/{job_id}'
ZIP_FILE = 'files.zip'

# Ensure folders exist
for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, ZIP_FOLDER]:
    if not os.path.exists(folder):
        os.makedirs(folder)


def clear_folders():
    """Clear content in specified folders."""
    for folder in [UPLOAD_FOLDER, PROCESSED_FOLDER, ZIP_FOLDER]:
        if os.path.exists(folder):
            shutil.rmtree(folder)
        os.makedirs(folder)

def generate_tree(path, prefix="", root_path=None):
    output = []
    path = Path(path)

    # Set the root path during the initial call
    if root_path is None:
        root_path = path

    # Compute the relative path from the root_path
    relative_path = path.relative_to(root_path)

    if path.is_dir():
        # Skip adding the root directory (which is the job ID directory)
        if str(relative_path) != '.':
            output.append(f"{prefix}|-- {path.name}/")
            prefix += "|   "
        for item in sorted(path.iterdir()):
            output.extend(generate_tree(item, prefix, root_path))
    elif path.is_file():
        output.append(f"{prefix}|-- {path.name}")

    return output




def zip_directory(source_dir, output_filename):
    with zipfile.ZipFile(output_filename, 'w', zipfile.ZIP_DEFLATED) as zipf:
        for root, _, files in os.walk(source_dir):
            for file in files:
                file_path = os.path.join(root, file)
                zipf.write(file_path, os.path.relpath(file_path, source_dir))

@app.route('/', methods=['GET'])
def index():
    # Clear folders at the start when the user opens the site
    clear_folders()
    return render_template('index.html')

from flask import Flask, render_template, request, send_file, redirect, Response, stream_with_context, url_for
import uuid

@app.route('/process', methods=['POST'])
def process():
    def generate():
        job_id = str(uuid.uuid4())

        # Create user-specific directories
        upload_folder = os.path.join('uploads', job_id)
        processed_folder = os.path.join('processed', job_id)
        zip_folder = os.path.join('zipped', job_id)
        zip_file = 'files.zip'

        for folder in [upload_folder, processed_folder, zip_folder]:
            os.makedirs(folder, exist_ok=True)

        # Handle uploaded files and directories
        if request.files:
            for key in request.files:
                file = request.files[key]

                # Use the full relative path provided by the file key
                relative_path = key.replace("\\", "/")
                file_path = os.path.join(upload_folder, relative_path)

                # Ensure the directory exists
                os.makedirs(os.path.dirname(file_path), exist_ok=True)

                # Save the file
                file.save(file_path)

        # Determine the input_directory to pass to run_full_pipeline_generator
        entries = os.listdir(upload_folder)
        if len(entries) == 1 and os.path.isdir(os.path.join(upload_folder, entries[0])):
            input_directory = os.path.join(upload_folder, entries[0])
        else:
            input_directory = upload_folder

        # Run the pipeline and yield status updates
        for status in run_full_pipeline_generator(input_directory, processed_folder, False):
            yield status + '\n'

        # Generate tree after processing
        tree_list = generate_tree(processed_folder)
        tree = "\n".join(tree_list)

        # Zip the processed files
        zip_output_path = os.path.join(zip_folder, zip_file)
        zip_directory(processed_folder, zip_output_path)

        # Clean up uploaded files
        shutil.rmtree(upload_folder)

        # Send the job ID and tree to the client
        yield f'PROCESSING_COMPLETE\n' + tree + f'\nJOB_ID:{job_id}'

    return Response(stream_with_context(generate()), mimetype='text/plain')


@app.route('/download', methods=['GET'])
def download():
    job_id = request.args.get('job_id')
    zip_output_path = os.path.join('zipped', job_id, 'files.zip')
    if os.path.exists(zip_output_path):
        return send_file(zip_output_path, as_attachment=True)
    else:
        return "No file available for download", 404

@app.route('/reset', methods=['GET'])
def reset():
    job_id = request.args.get('job_id')
    # Clean up user-specific directories
    for folder in ['uploads', 'processed', 'zipped']:
        job_folder = os.path.join(folder, job_id)
        if os.path.exists(job_folder):
            shutil.rmtree(job_folder)
    return redirect('/')


def cleanup_old_jobs():
    while True:
        current_time = time.time()
        for folder in ['uploads', 'processed', 'zipped']:
            for job_folder in os.listdir(folder):
                folder_path = os.path.join(folder, job_folder)
                if os.path.isdir(folder_path):
                    creation_time = os.path.getctime(folder_path)
                    # Delete folders older than 1 hour
                    if current_time - creation_time > 3600:
                        shutil.rmtree(folder_path)
        time.sleep(3600)  # Run cleanup every hour

# Start the cleanup thread
cleanup_thread = threading.Thread(target=cleanup_old_jobs)
cleanup_thread.daemon = True
cleanup_thread.start()


if __name__ == '__main__':
    app.run(debug=False)
