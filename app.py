from flask import Flask, render_template, request
from src.pipeline.predict_pipeline import CustomData, PredictPipeline


app = Flask(__name__)

@app.route('/')
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
def predict_data():
    if request.method == "GET":
        return render_template('index.html')
    else:
        data = CustomData(
            N= int(request.form.get('N')),
            P= int(request.form.get('P')),
            K= int(request.form.get('K')),
            temperature= float(request.form.get('temperature')),
            humidity= float(request.form.get('humidity')),
            ph= float(request.form.get('ph')),
            rainfall= float(request.form.get('rainfall'))
        )

        pred_df = data.get_data_as_dataframe()
        print(pred_df)

        predict_pipeline = PredictPipeline()
        results = predict_pipeline.predict(pred_df)

        return render_template('index.html', results=results[0])
    
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=8000)