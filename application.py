from flask import Flask, render_template, request
import csv
import io
import base64
import os
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score

# Changed 'app' to 'application'
application = Flask(__name__)

UPLOAD_FOLDER = 'uploads'
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
application.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

@application.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        
        # STEP 1: Handle File Upload
        if 'file' in request.files:
            file = request.files['file']
            if file.filename == '':
                return render_template('index.html', error_msg='No selected file.', step=1)
            
            try:
                filepath = os.path.join(application.config['UPLOAD_FOLDER'], 'data.csv')
                file.save(filepath)
                
                with open(filepath, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    headers = next(csv_reader, None)
                    
                    if not headers:
                        raise ValueError("The uploaded CSV file is empty.")
                    
                    data_preview = [row for i, row in enumerate(csv_reader) if i < 10]
                    
                return render_template('index.html', headers=headers, data_preview=data_preview, step=2)
                
            except Exception as e:
                return render_template('index.html', error_msg=f"Error reading file: {e}", step=1)

        # STEP 2: Process Multiple Variables (MLR)
        elif 'x_vars' in request.form and 'y_var' in request.form:
            x_vars = request.form.getlist('x_vars') # Gets multiple selected values
            y_var = request.form['y_var']
            filepath = os.path.join(application.config['UPLOAD_FOLDER'], 'data.csv')
            
            try:
                with open(filepath, 'r', encoding='utf-8') as f:
                    csv_reader = csv.reader(f)
                    headers = list(next(csv_reader))
                    
                    x_indices = [headers.index(v) for v in x_vars]
                    y_idx = headers.index(y_var)
                    
                    x_data = []
                    y_data = []
                    
                    for row in csv_reader:
                        try:
                            # Extract multiple X values for the current row
                            x_row = [float(row[i].strip()) for i in x_indices]
                            y_val = float(row[y_idx].strip())
                            x_data.append(x_row)
                            y_data.append(y_val)
                        except (ValueError, IndexError):
                            continue 
                
                X = np.array(x_data)
                y = np.array(y_data)
                
                if len(X) == 0:
                    raise ValueError("No valid numeric data rows found.")

                # MLR Model
                model = LinearRegression()
                model.fit(X, y)
                y_pred = model.predict(X)
                accuracy = round(r2_score(y, y_pred) * 100, 2)
                
                # Coefficients for display
                coef_map = dict(zip(x_vars, [round(c, 4) for c in model.coef_]))
                intercept = round(model.intercept_, 4)

                # Generate "Actual vs Predicted" Plot
                plt.figure(figsize=(8, 6))
                plt.scatter(y, y_pred, color='teal', alpha=0.6, label='Data points')
                # Ideal prediction line (y=x)
                plt.plot([y.min(), y.max()], [y.min(), y.max()], 'r--', lw=2, label='Perfect Fit')
                
                plt.xlabel(f'Actual {y_var}')
                plt.ylabel(f'Predicted {y_var}')
                plt.title(f'MLR: Predicted vs Actual (R²: {accuracy}%)')
                plt.legend()
                plt.grid(True)
                
                img = io.BytesIO()
                plt.savefig(img, format='png')
                img.seek(0)
                plot_url = base64.b64encode(img.getvalue()).decode('utf8')
                plt.close()
                
                return render_template('index.html', plot_url=plot_url, accuracy=accuracy, 
                                       coefs=coef_map, intercept=intercept, step=3)
                
            except Exception as e:
                return render_template('index.html', error_msg=f"Analysis error: {e}", step=1)

    return render_template('index.html', step=1)

if __name__ == '__main__':
    application.run(debug=True, port=5000)