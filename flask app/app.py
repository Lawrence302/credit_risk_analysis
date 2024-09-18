from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model
model = pickle.load(open('model/lr_model_fs.pkl', 'rb'))

app = Flask(__name__)




@app.route('/')
def home():
    return render_template('home.html')

@app.route('/form')
def form():
    return render_template('form.html')


@app.route('/submit', methods=['POST'])
def submit():
    # Retrieve form data
    data = {
        'Percent_income': request.form.get('Percent_income'),
        'Default': 1 if request.form.get('Default')== 'Yes' else 0,
        'Cred_length': request.form.get('Cred_length'),
        'Rate': request.form.get('Rate'),
    }

    # Handle Home field
    home_options = ['MORTGAGE', 'OTHER', 'OWN', 'RENT']
    selected_home = request.form.get('Home')
    for option in home_options:
        data[f'Home_{option}'] = 1 if selected_home == option else 0

        # Handle Intent field

    intent_options = ['DEBTCONSOLIDATION', 'EDUCATION', 'HOMEIMPROVEMENT', 'MEDICAL', 'PERSONAL', 'VENTURE']
    selected_intent = request.form.get('Intent')
    for option in intent_options:
        data[f'Intent_{option}'] = 1 if selected_intent == option else 0

    # convert data to dataframe
    df = pd.DataFrame([data])
    print(df)
    # Make prediction
    prediction = model.predict(df)
    print(prediction)
    

    return f"Form submitted successfully! Data: <pre>{df.to_html()}</pre> <br> Prediction: {prediction}"



    # return f"Form submitted successfully! Data: {data}"





if __name__ == '__main__':
    app.run(debug=True)