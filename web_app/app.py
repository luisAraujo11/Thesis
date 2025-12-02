"""
Neuroimaging Pipeline Web Application
-----------------------------------
A Flask web application for processing neuroimaging data.
Handles DICOM and NIFTI formats with validation and error handling.
"""

import os
import json
import secrets
import logging
import nibabel as nib
from pathlib import Path
from datetime import datetime
from werkzeug.utils import secure_filename
from werkzeug.security import generate_password_hash, check_password_hash
from flask import Flask, render_template, request, redirect, url_for, flash, send_file, session
from flask_login import LoginManager, UserMixin, login_user, login_required, logout_user, current_user

# Import the pipeline functionality
from pipeline import run_pipeline

# Initialize Flask application
app = Flask(__name__)

# Application Configuration
app.config.update(
    SECRET_KEY=os.getenv('SECRET_KEY', secrets.token_hex(32)),
    UPLOAD_FOLDER=Path('uploads'),
    OUTPUT_FOLDER=Path('outputs'),
    MAX_CONTENT_LENGTH=1024 * 1024 * 1024,  # 1GB max file size
    SESSION_PERMANENT=False,
    PORT=int(os.getenv('PORT', 37134))
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
def load_users():
    """
    Load users from config.json if it exists.
    Otherwise, use default admin user with password '54321'.
    
    Returns:
        dict: {username: hashed_password}
    """
    config_file = Path('config.json')
    if config_file.exists():
        with open(config_file) as f:
            users_data = json.load(f).get('users', {})
            logger.info(f"Loaded {len(users_data)} users from config.json")
            return users_data
    
    # Default fallback: admin with password 54321
    logger.info("No config.json found, using default admin user")
    return {'admin': generate_password_hash('54321')}

users = load_users()

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

@app.route('/upload', methods=['GET', 'POST'])
@login_required
def upload():
    """Handle file uploads and process neuroimaging data."""
    if request.method == 'POST':
        processing_logger = None
        try:
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            upload_dir = app.config['UPLOAD_FOLDER'] / f"upload_{timestamp}"
            output_dir = app.config['OUTPUT_FOLDER']
            upload_dir.mkdir(parents=True, exist_ok=True)
            output_dir.mkdir(exist_ok=True)
            
            processing_logger, log_file = get_processing_logger(timestamp)
            processing_logger.info(f"Starting upload processing: {timestamp}")
            
            submit_type = request.form.get('submit')
            
            if submit_type == 'folder':
                if 'folder' not in request.files:
                    flash("No folder selected", "error")
                    return redirect(url_for('upload'))
                
                files = request.files.getlist('folder')
                if not files or files[0].filename == '':
                    flash("Empty folder submitted", "error")
                    return redirect(url_for('upload'))
                
                file_count = 0
                for file in files:
                    if file.filename:
                        filepath = upload_dir / file.filename
                        filepath.parent.mkdir(parents=True, exist_ok=True)
                        file.save(str(filepath))
                        file_count += 1
                
                processing_logger.info(f"Saved {file_count} files from folder")
                
                # Process files
                output_files, analysis = run_pipeline(
                    upload_dir,
                    output_dir,
                    logger=processing_logger
                )
                
                if output_files and analysis:
                    processing_logger.info("Processing completed successfully")
                    session['last_processing_time'] = timestamp
                    session['last_log_file'] = str(log_file)
                    flash(f'Processed {len(output_files)} files successfully!', 'success')
                    return redirect(url_for('results'))
                else:
                    flash("Upload failed: Invalid brain MRI files.", "error")
                    return redirect(url_for('upload'))
                    
            elif submit_type == 'file':
                    # Single file upload validation
                    if 'file' not in request.files:
                        flash("No file in request", "error")
                        return redirect(url_for('upload'))
                    
                    file = request.files['file']
                    if file.filename == '':
                        flash("No file selected", "error")
                        return redirect(url_for('upload'))
                    
                    # Check file extension
                    file_extension = Path(file.filename).suffix.lower()
                    if file_extension not in ['.nii', '.gz']:
                        flash(f"Unsupported file: {file_extension}. Use .nii or .nii.gz", "error")
                        return redirect(url_for('upload'))
                    
                    # Handle NIFTI file
                    filename = secure_filename(file.filename)
                    output_path = output_dir / filename
                    file.save(str(output_path))
                    processing_logger.info(f"Saved NIFTI: {output_path}")
                    
                    # Store processing information
                    session['last_processing_time'] = timestamp
                    session['last_log_file'] = str(log_file)
                    
                    flash(f'Uploaded {filename} successfully!', 'success')
                    return redirect(url_for('results'))
            
            else:
                flash("Invalid upload type", "error")
                return redirect(url_for('upload'))
                
        except Exception as e:
            app.logger.exception("Upload error")
            flash(f"Upload failed: {str(e)}", "error")
            return redirect(url_for('upload'))
            
        finally:
            # Clean up logging handlers
            if processing_logger:
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
    output_dir = app.config['OUTPUT_FOLDER']
    
    if not output_dir.exists():
        return render_template('results.html', modality_groups={})
    
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
        file_path = app.config['OUTPUT_FOLDER'] / filename
        if not file_path.exists():
            flash('File not found', 'error')
            return redirect(url_for('results'))
        return send_file(str(file_path), as_attachment=True)
    except Exception as e:
        logger.error(f"Error downloading file {filename}: {str(e)}")
        flash('Error downloading file', 'error')
        return redirect(url_for('results'))

@app.route('/nifti_viewer/<path:file_path>')
@login_required
def nifti_viewer(file_path):
    """Display NIFTI file viewer"""
    try:
        nifti_path = app.config['OUTPUT_FOLDER'] / file_path
        
        if not nifti_path.exists():
            flash('NIFTI file not found', 'error')
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
        flash('Error loading NIFTI viewer', 'error')
        return redirect(url_for('results'))

@app.route('/serve_nifti/<path:filepath>')
@login_required
def serve_nifti(filepath):
    """Serve NIFTI file for Papaya viewer"""
    try:
        file_path = app.config['OUTPUT_FOLDER'] / filepath
        
        if not file_path.exists():
            return "File not found", 404

        # Simple file serving - keep it basic
        return send_file(str(file_path), mimetype='application/octet-stream')

    except Exception as e:
        app.logger.exception("Error serving NIFTI file")
        return str(e), 500

# Application initialization
if __name__ == '__main__':
    # Create required directories
    for directory in ['uploads', 'outputs', 'logs']:
        Path(directory).mkdir(exist_ok=True)
    
    # Start the Flask application
    port = 80
    print(f"Starting Flask server on port {port}...")
    print(f"Access at: http://gpuserver.di.uminho.pt:{port}/")
    
    # Debug: print actual bind address
    print(f"Binding to 0.0.0.0:{port}")
    print(f"Internal IP will be: {os.popen('hostname -I').read().strip()}")
    
    app.run(
        host='0.0.0.0',
        port=port,
        debug=False,
        use_reloader=False
    )