"""
Flask Frontend Application for AI Data Cleaning System
Replaces Streamlit with a modern web interface
"""

from flask import Flask, render_template, request, jsonify, send_file, session
from flask_cors import CORS
import requests
import pandas as pd
import io
import os
from werkzeug.utils import secure_filename
from dotenv import load_dotenv
import json

# Load environment variables
load_dotenv()

# Initialize Flask app
app = Flask(__name__)
app.secret_key = os.getenv("FLASK_SECRET_KEY", "your-secret-key-change-in-production")
CORS(app)

# Register ML routes blueprint
from ml_routes import ml_bp
app.register_blueprint(ml_bp)

# Configuration
UPLOAD_FOLDER = "uploads"
ALLOWED_EXTENSIONS = {'csv', 'xlsx', 'xls'}
FASTAPI_URL = f"http://{os.getenv('FASTAPI_HOST', '127.0.0.1')}:{os.getenv('FASTAPI_PORT', '8000')}"

# Create upload folder if it doesn't exist
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size


def allowed_file(filename):
    """Check if file extension is allowed"""
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    """Main page - Data source selection"""
    return render_template('index.html')


@app.route('/upload')
def upload_page():
    """File upload page"""
    return render_template('upload.html')


@app.route('/database')
def database_page():
    """Database query page"""
    return render_template('database.html')


@app.route('/api_data')
def api_data_page():
    """API data fetch page"""
    return render_template('api_data.html')


@app.route('/ml_dashboard')
def ml_dashboard_page():
    """ML Training Dashboard"""
    return render_template('ml_dashboard.html')


@app.route('/inference')
def inference_page():
    """Model Inference page"""
    return render_template('inference.html')


@app.route('/tuning')
def tuning_page():
    """Hyperparameter Tuning page"""
    return render_template('tuning.html')


@app.route('/model_versions')
def model_versions_page():
    """Model Version Management page"""
    return render_template('model_versions.html')


@app.route('/observability')
def observability_page():
    """Observability Dashboard page"""
    return render_template('observability.html')


@app.route('/api/clean-file', methods=['POST'])
def clean_file():
    """
    Handle file upload and cleaning
    Endpoint for CSV/Excel file processing
    """
    try:
        # Check if file is present
        if 'file' not in request.files:
            return jsonify({'error': 'No file provided'}), 400
        
        file = request.files['file']
        
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        if not allowed_file(file.filename):
            return jsonify({'error': 'Invalid file type. Use CSV or Excel'}), 400
        
        # Read file into DataFrame for preview
        file_extension = file.filename.rsplit('.', 1)[1].lower()
        
        if file_extension == 'csv':
            df_raw = pd.read_csv(file)
        else:
            df_raw = pd.read_excel(file)
        
        # Reset file pointer
        file.seek(0)
        
        # Send to FastAPI backend for cleaning
        files = {'file': (file.filename, file.read(), file.content_type)}
        response = requests.post(f"{FASTAPI_URL}/clean-data", files=files)
        
        if response.status_code == 200:
            result = response.json()
            cleaned_data = result.get('cleaned_data', [])
            
            # Save cleaned dataset to registry for ML training
            dataset_id = None
            if cleaned_data:
                try:
                    from ml_routes import DatasetRegistry
                    df_cleaned = pd.DataFrame(cleaned_data)
                    meta = DatasetRegistry.save(
                        df_cleaned, 
                        source='cleaned_upload',
                        metadata={
                            'original_filename': file.filename,
                            'raw_rows': len(df_raw),
                            'cleaned_rows': len(df_cleaned)
                        }
                    )
                    dataset_id = meta['id']
                except Exception as e:
                    print(f"Failed to save to dataset registry: {e}")
            
            return jsonify({
                'success': True,
                'raw_data': df_raw.head(100).to_dict(orient='records'),
                'cleaned_data': cleaned_data[:100] if len(cleaned_data) > 100 else cleaned_data,
                'raw_shape': df_raw.shape,
                'cleaned_shape': [len(cleaned_data), len(cleaned_data[0].keys()) if cleaned_data else 0],
                'message': 'Data cleaned successfully!',
                'dataset_id': dataset_id,
                'can_train': dataset_id is not None
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to clean data. Backend error.'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing file: {str(e)}'
        }), 500



@app.route('/api/clean-database', methods=['POST'])
def clean_database():
    """
    Handle database query and cleaning
    """
    try:
        data = request.get_json()
        db_url = data.get('db_url')
        query = data.get('query')
        
        if not db_url or not query:
            return jsonify({'error': 'Database URL and query are required'}), 400
        
        # Send to FastAPI backend
        response = requests.post(
            f"{FASTAPI_URL}/clean-db",
            json={'db_url': db_url, 'query': query}
        )
        
        if response.status_code == 200:
            result = response.json()
            cleaned_data = result.get('cleaned_data', [])
            
            return jsonify({
                'success': True,
                'cleaned_data': cleaned_data[:100] if len(cleaned_data) > 100 else cleaned_data,
                'total_rows': len(cleaned_data),
                'message': 'Database data cleaned successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch or clean database data.'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing database query: {str(e)}'
        }), 500


@app.route('/api/clean-api-data', methods=['POST'])
def clean_api_data():
    """
    Handle API data fetch and cleaning
    """
    try:
        data = request.get_json()
        api_url = data.get('api_url')
        
        if not api_url:
            return jsonify({'error': 'API URL is required'}), 400
        
        # Send to FastAPI backend
        response = requests.post(
            f"{FASTAPI_URL}/clean-api",
            json={'api_url': api_url}
        )
        
        if response.status_code == 200:
            result = response.json()
            cleaned_data = result.get('cleaned_data', [])
            
            return jsonify({
                'success': True,
                'cleaned_data': cleaned_data[:100] if len(cleaned_data) > 100 else cleaned_data,
                'total_rows': len(cleaned_data),
                'message': 'API data cleaned successfully!'
            })
        else:
            return jsonify({
                'success': False,
                'error': 'Failed to fetch or clean API data.'
            }), 500
            
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error processing API data: {str(e)}'
        }), 500


@app.route('/api/download-csv', methods=['POST'])
def download_csv():
    """
    Download cleaned data as CSV
    """
    try:
        data = request.get_json()
        cleaned_data = data.get('data', [])
        
        if not cleaned_data:
            return jsonify({'error': 'No data to download'}), 400
        
        # Convert to DataFrame
        df = pd.DataFrame(cleaned_data)
        
        # Create CSV in memory
        output = io.StringIO()
        df.to_csv(output, index=False)
        output.seek(0)
        
        # Create BytesIO for file download
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            mimetype='text/csv',
            as_attachment=True,
            download_name='cleaned_data.csv'
        )
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': f'Error creating download: {str(e)}'
        }), 500


@app.route('/api/health', methods=['GET'])
def health_check():
    """Health check endpoint"""
    try:
        # Check FastAPI backend
        response = requests.get(f"{FASTAPI_URL}/", timeout=5)
        backend_status = response.status_code == 200
    except:
        backend_status = False
    
    return jsonify({
        'flask': 'healthy',
        'fastapi_backend': 'healthy' if backend_status else 'unavailable',
        'status': 'ok' if backend_status else 'degraded'
    })


@app.errorhandler(413)
def too_large(e):
    """Handle file too large error"""
    return jsonify({'error': 'File is too large. Maximum size is 16MB'}), 413


@app.errorhandler(404)
def not_found(e):
    """Handle 404 errors"""
    return render_template('404.html'), 404


@app.errorhandler(500)
def server_error(e):
    """Handle 500 errors"""
    return jsonify({'error': 'Internal server error'}), 500


if __name__ == '__main__':
    host = os.getenv('FLASK_HOST', '127.0.0.1')
    port = int(os.getenv('FLASK_PORT', 5000))
    debug = os.getenv('FLASK_DEBUG', 'True').lower() == 'true'
    
    print(f"""
    ╔════════════════════════════════════════════════════════════╗
    ║  🧹 AI Data Cleaning System - Flask Frontend              ║
    ╠════════════════════════════════════════════════════════════╣
    ║  Flask UI:     http://{host}:{port}                    ║
    ║  Backend API:  {FASTAPI_URL}                              ║
    ║  Status:       Starting...                                 ║
    ╚════════════════════════════════════════════════════════════╝
    """)
    
    app.run(host=host, port=port, debug=debug)
