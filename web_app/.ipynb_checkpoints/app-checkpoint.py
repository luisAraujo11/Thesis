"""
Neuroimaging Pipeline Web Application
-----------------------------------
A Flask web application that provides an interface for processing neuroimaging data.
Handles both DICOM and NIFTI file formats with proper validation and error handling.

Author: Luís Araujo
Date: December 2024 - 2025
"""

from flask import Flask, render_template, request, redirect, url_for, flash, send_file, jsonify, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
import os
import json
import shutil
from pathlib import Path
import secrets
from datetime import datetime
import logging
import nibabel as nib
import numpy as np
import subprocess

# Import the pipeline functionality
from pipeline import run_pipeline

# Initialize Flask application
app = Flask(__name__)

# Application Configuration
app.config.update(
    SECRET_KEY=secrets.token_hex(32),
    UPLOAD_FOLDER='uploads',
    OUTPUT_FOLDER='outputs',
    MAX_CONTENT_LENGTH=1024 * 1024 * 1024,  # 1GB max file size
    SESSION_PERMANENT=False
)

@app.template_filter('format_timestamp')
def format_timestamp(timestamp):
    """Convert timestamp to readable format"""
    from datetime import datetime
    dt = datetime.fromtimestamp(timestamp)
    return dt.strftime('%Y-%m-%d %H:%M:%S')

# Configure logging system
def setup_logging():
    """
    Configure unified logging system for web application and pipeline processing.
    Creates a single 'logs' directory with:
    1. webapp.log - General application logs
    2. Individual processing run logs stored in dated subdirectories
    """
    log_dir = Path('logs')
    log_dir.mkdir(exist_ok=True)
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s in %(module)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Web application file handler
    webapp_handler = logging.FileHandler(log_dir / 'webapp.log')
    webapp_handler.setFormatter(formatter)
    webapp_handler.setLevel(logging.INFO)
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setFormatter(formatter)
    console_handler.setLevel(logging.INFO)
    
    # Configure root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.INFO)
    root_logger.handlers = []  # Remove existing handlers
    root_logger.addHandler(webapp_handler)
    root_logger.addHandler(console_handler)
    
    # Configure Flask logger
    app.logger.setLevel(logging.INFO)
    
    return root_logger

def get_processing_logger(timestamp):
    """
    Create a logger for a specific processing run.
    
    Args:
        timestamp (str): Timestamp identifier for the processing run
        
    Returns:
        tuple: (logger, log_file_path)
    """
    log_dir = Path('logs') / timestamp
    log_dir.mkdir(exist_ok=True)
    
    log_file = log_dir / 'processing.log'
    
    logger = logging.getLogger(f'processing.{timestamp}')
    logger.setLevel(logging.INFO)
    logger.handlers = []  # Remove existing handlers
    
    formatter = logging.Formatter(
        '[%(asctime)s] %(levelname)s: %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    file_handler.setLevel(logging.INFO)
    
    logger.addHandler(file_handler)
    
    return logger, log_file
    
# Initialize logging when app starts
logger = setup_logging()

# Login manager setup
login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'login'

# User management
class User(UserMixin):
    def __init__(self, id):
        self.id = id

class ValidationError(Exception):
    """Custom exception for input validation errors"""
    pass

# Simple user database - in production, use a real database
users = {
    'admin': generate_password_hash('54321')
}

@login_manager.user_loader
def load_user(user_id):
    if user_id not in users:
        return None
    return User(user_id)

# Routes
@app.route('/')
def home():
    """Home page route"""
    if current_user.is_authenticated:
        return redirect(url_for('upload'))
    return redirect(url_for('login'))

@app.route('/login', methods=['GET', 'POST'])
def login():
    """Login page route"""
    if request.method == 'POST':
        username = request.form.get('username')
        password = request.form.get('password')
        
        if username in users and check_password_hash(users[username], password):
            user = User(username)
            login_user(user)
            logger.info(f"Successful login for user: {username}")
            return redirect(url_for('upload'))
        
        logger.warning(f"Failed login attempt for username: {username}")
        flash('Invalid username or password')
    return render_template('login.html')

@app.route('/logout')
@login_required
def logout():
    """Logout route"""
    logout_user()
    return redirect(url_for('login'))

# In app.py, update the upload route's error handling:

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Handle file uploads and process neuroimaging data."""
    if request.method == 'POST':
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            upload_dir = Path(app.config['UPLOAD_FOLDER']) / f"upload_{timestamp}"
            output_dir = Path(app.config['OUTPUT_FOLDER'])
            upload_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(exist_ok=True)
            
            processing_logger, log_file = get_processing_logger(timestamp)
            app.logger.info(f"Starting new upload processing: {timestamp}")
            processing_logger.info("Beginning processing run")
            
            submit_type = request.form.get('submit')
            processing_logger.info(f"Upload type: {submit_type}")
            
            if submit_type == 'folder':
                if 'folder' not in request.files:
                    flash("No folder selected")
                    return redirect(url_for('upload'))
                
                files = request.files.getlist('folder')
                if not files or files[0].filename == '':
                    flash("Empty folder submitted")
                    return redirect(url_for('upload'))
                
                file_count = 0
                for file in files:
                    if file.filename:
                        filepath = upload_dir / file.filename
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        file.save(str(filepath))
                        file_count += 1
                
                processing_logger.info(f"Saved {file_count} files from uploaded folder")
                
                # Process files
                output_files, analysis = run_pipeline(
                    upload_dir,
                    output_dir,
                    logger=processing_logger
                )
                
                if output_files and analysis:
                    processing_logger.info("Directory processing completed successfully")
                    session['last_processing_time'] = timestamp
                    session['last_log_file'] = str(log_file)
                    flash('Directory processing completed successfully!')
                    return redirect(url_for('results'))
                else:
                    flash("Upload failed: The uploaded files are not valid brain MRI files.", "error")
                    return redirect(url_for('upload'))
                    
            elif submit_type == 'file':
                    # Single file upload validation
                    if 'file' not in request.files:
                        raise ValidationError("No file in request")
                    
                    file = request.files['file']
                    if file.filename == '':
                        raise ValidationError("No file selected")
                    
                    # Check file extension
                    file_extension = Path(file.filename).suffix.lower()
                    if file_extension not in ['.nii', '.gz']:
                        raise ValidationError(f"Unsupported file type: {file_extension}. Only .nii or .nii.gz files are accepted.")
                    
                    # Handle NIFTI file
                    filename = secure_filename(file.filename)
                    output_path = output_dir / filename
                    file.save(str(output_path))
                    processing_logger.info(f"Saved NIFTI file to output: {output_path}")
                    
                    # Store processing information
                    session['last_processing_time'] = timestamp
                    session['last_log_file'] = str(log_file)
                    
                    flash('NIFTI file uploaded successfully!')
                    return redirect(url_for('results'))
            
            else:
                flash("Invalid upload type")
                return redirect(url_for('upload'))
                
        except Exception as e:
            flash(str(e))
            return redirect(url_for('upload'))
            
        finally:
            # Clean up logging handlers
            if 'processing_logger' in locals():
                for handler in processing_logger.handlers[:]:
                    processing_logger.removeHandler(handler)
                    handler.close()
    
    return render_template('upload.html')

@app.route('/results')
@login_required
def results():
    """
    Organize and display neuroimaging results grouped by modality.
    Each modality will have its NIFTI and associated metadata including sequence name.
    """
    output_dir = Path(app.config['OUTPUT_FOLDER'])
    
    # Dictionary to store files grouped by modality
    modality_groups = {}
    
    # Scan output directory
    for item in output_dir.iterdir():
        if item.is_file() and item.suffix.lower() in {'.gz', '.nii', '.json'}:
            try:
                # Get base name
                base_name = item.name
                if base_name.endswith('.nii.gz'):
                    base_name = base_name[:-7]
                elif base_name.endswith('.nii'):
                    base_name = base_name[:-4]
                elif base_name.endswith('.json'):
                    base_name = base_name[:-5]
                
                # Get sequence name from JSON if available
                if item.suffix.lower() == '.json':
                    with open(item) as f:
                        metadata = json.load(f)
                        sequence_name = metadata.get('series_description', base_name)
                else:
                    json_file = item.with_suffix('.json')
                    if json_file.exists():
                        with open(json_file) as f:
                            metadata = json.load(f)
                            sequence_name = metadata.get('series_description', base_name)
                    else:
                        sequence_name = base_name
                
            except:
                sequence_name = base_name = item.stem
            
            # Initialize or update modality group
            if base_name not in modality_groups:
                modality_groups[base_name] = {
                    'nifti': None,
                    'json': None,
                    'timestamp': item.stat().st_mtime,
                    'name': sequence_name
                }
            
            # Store file path based on type
            if item.suffix.lower() in {'.gz', '.nii'}:
                modality_groups[base_name]['nifti'] = item.name
            elif item.suffix.lower() == '.json':
                modality_groups[base_name]['json'] = item.name
    
    # Sort by timestamp
    sorted_groups = dict(sorted(
        modality_groups.items(),
        key=lambda x: x[1]['timestamp'],
        reverse=True
    ))
    
    return render_template(
        'results.html',
        modality_groups=sorted_groups,
        last_processing_time=session.get('last_processing_time')
    )

@app.route('/download/<path:filename>')
@login_required
def download(filename):
    """File download route"""
    try:
        file_path = Path(app.config['OUTPUT_FOLDER']) / filename
        return send_file(str(file_path), as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        flash('Error downloading file')
        return redirect(url_for('results'))

@app.route('/download_folder/<path:folder_name>') # pra ja n esta a ser usado
@login_required
def download_folder(folder_name):
    """Folder download route"""
    try:
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        folder_path = output_dir / folder_name
        
        if not folder_path.exists():
            flash('Folder not found')
            return redirect(url_for('results'))
        
        # Create ZIP file
        zip_filename = f"{folder_name}.zip"
        zip_path = output_dir / zip_filename
        
        shutil.make_archive(
            str(zip_path.with_suffix('')),
            'zip',
            folder_path
        )
        
        return send_file(
            str(zip_path),
            as_attachment=True,
            download_name=zip_filename
        )
    except Exception as e:
        logger.error(f"Error downloading folder {folder_name}: {str(e)}")
        flash('Error downloading folder')
        return redirect(url_for('results'))

@app.route('/nifti_viewer/<path:file_path>')
@login_required
def nifti_viewer(file_path):
    """Display NIFTI file viewer"""
    try:
        output_dir = Path(app.config['OUTPUT_FOLDER'])
        nifti_path = output_dir / file_path
        
        if not nifti_path.exists():
            flash('NIFTI file not found')
            return redirect(url_for('results'))
            
        # Load NIFTI metadata for display
        img = nib.load(str(nifti_path))
        metadata = {
            'dimensions': f"{img.shape}",
            'voxel_size': f"{img.header.get_zooms()}",
            'data_type': f"{img.get_data_dtype()}",
            'orientation': f"{nib.aff2axcodes(img.affine)}"
        }
        
        # Look for associated JSON metadata file
        json_path = nifti_path.with_suffix('.json')
        if json_path.exists():
            with open(json_path) as f:
                metadata.update(json.load(f))
        
        return render_template('nifti_viewer.html',
                            file_path=file_path,
                            nifti_url=url_for('serve_nifti', filepath=file_path),
                            metadata=metadata)
                            
    except Exception as e:
        app.logger.error(f"Error viewing NIFTI {file_path}: {str(e)}")
        flash('Error loading NIFTI viewer')
        return redirect(url_for('results'))

@app.route('/serve_nifti/<path:filepath>')
@login_required
def serve_nifti(filepath):
    """Serve NIFTI file for Papaya viewer"""
    try:
        file_path = Path(app.config['OUTPUT_FOLDER']) / filepath
        
        if not file_path.exists():
            return "File not found", 404

        # Simple file serving - keep it basic
        return send_file(
            str(file_path),
            mimetype='application/octet-stream'
        )

    except Exception as e:
        app.logger.exception("Error serving NIFTI file")
        return str(e), 500

# Application initialization
if __name__ == '__main__':
    # Create required directories
    for directory in ['uploads', 'outputs', 'logs']:
        Path(directory).mkdir(exist_ok=True)
    
    # Start the Flask application
    print("Starting Flask server...")
    print("Access the web interface at: http://gpuserver.di.uminho.pt:37134/")
    app.run(
        host='0.0.0.0',
        port=80,
        debug=False,
        use_reloader=False
    )