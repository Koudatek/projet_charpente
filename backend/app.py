from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import os
from truss_analyzer import TrussAnalyzer

app = Flask(__name__)
CORS(app)

STATIC_DIR = os.path.join(os.path.dirname(__file__), 'static')

@app.route('/')
def index():
    return 'Backend projet-charpente opérationnel !'

@app.route('/calculate', methods=['POST'])
def calculate():
    params = request.get_json()
    analyzer = TrussAnalyzer()
    try:
        results = analyzer.run_analysis(params)
        return jsonify(results)
    except Exception as e:
        return jsonify({"error": str(e)}), 400

@app.route('/generate_pdf', methods=['POST'])
def generate_pdf():
    params = request.get_json()
    print('PARAMS REÇUS POUR PDF:', params)
    analyzer = TrussAnalyzer()
    try:
        analyzer.run_analysis(params)
        pdf_path = analyzer.generate_pdf_report()
        filename = os.path.basename(pdf_path)
        return send_from_directory(STATIC_DIR, filename, as_attachment=True)
    except Exception as e:
        print('ERREUR PDF:', e)
        return jsonify({"error": str(e)}), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port) 