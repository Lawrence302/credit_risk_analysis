from flask import Flask, render_template, request
import pandas as pd
import pickle

# Load model
model = pickle.load(open('model/lr_model_fs.pkl', 'rb'))

app = Flask(__name__)

# The expected feature order for the model
# Define the expected feature order
expected_feature_order = [
    'Percent_income', 'Default', 'Cred_length', 
    'Intent_DEBTCONSOLIDATION', 'Intent_EDUCATION', 
    'Intent_HOMEIMPROVEMENT', 'Intent_MEDICAL', 
    'Intent_PERSONAL', 'Intent_VENTURE', 
    'Home_MORTGAGE', 'Home_OTHER', 
    'Home_OWN', 'Home_RENT', 
    'Rate'
]


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


    # just experimengin 
     # Create a second DataFrame for filled fields
    filled_fields = { 
        'Home': [],
        'Intention': [],
        'Percent_income': request.form.get('Percent_income'),
        'Default': 1 if request.form.get('Default')== 'Yes' else 0,
        'Cred_length': request.form.get('Cred_length'),
        'Rate': request.form.get('Rate'),

    }
    
    # Add Home fields
    for option in home_options:
        if data[f'Home_{option}'] == 1:
            filled_fields['Home'].append(option)
           

    # Add Intent fields
    for option in intent_options:
        if data[f'Intent_{option}'] == 1:
            filled_fields['Intention'].append(option)

            

    # Convert filled fields to DataFrame
    filled_df = pd.DataFrame(filled_fields)


    # convert data to dataframe
    df = pd.DataFrame([data])

    # Here i need to re-order the columns to match the expected feature order
    df = df[expected_feature_order]
    
    # Make prediction
    prediction = model.predict(df)
    prediction_proba = model.predict_proba(df)
   
     # Round probabilities to two decimal places
    risk_prob = round(prediction_proba[0][1] * 100, 2)  # Probability of risk
    no_risk_prob = round(prediction_proba[0][0] * 100, 2)  # Probability of no risk

    # Create messages
    risk_message = f"The probability of risk is: {risk_prob}%"
    no_risk_message = f"The probability of no risk is: {no_risk_prob}%"

    # return f"Form submitted successfully! Data: <pre>{df.to_html()}</pre> <br> Prediction: {prediction} {model.predict_proba(df)}"
     # Determine risk level
    
    if prediction[0] == 1:
        risk_level = "Based on the provided data, granting this loan might be high risk. "
    else:
        risk_level = "Based on the provided data, granting this loan appears to be low risk. "

    # return render_template('result.html', data=df.to_html(classes='data'), prediction=prediction_proba, risk_level=risk_level)
    return render_template(
        'result.html', 
        data=df.to_html(classes='data'), 
        filled_data=filled_df.to_html(classes='filled-data'),  # New table for filled fields
        # prediction=prediction_proba, 
        risk_level=risk_level,
        risk_message=risk_message,
        no_risk_message=no_risk_message
    )



@app.route('/about')
def about():
    return render_template('about.html')




if __name__ == '__main__':
    app.run(debug=True)