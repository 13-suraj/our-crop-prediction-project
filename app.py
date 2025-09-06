from flask import Flask, render_template, request
from flask_cors import CORS, cross_origin
from src.pipeline.predict_pipeline import CustomData, PredictPipeline

app = Flask(__name__)

@app.route('/')
@cross_origin()
def home_page():
    return render_template('index.html')

@app.route('/predict', methods=["GET", "POST"])
@cross_origin()
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
    print("About to start Flask server...")
    print("Flask is running at http://127.0.0.1:5000/")
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
