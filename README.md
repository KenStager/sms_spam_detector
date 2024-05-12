Here's a detailed README file that you can use for your GitHub repository:

---

# SMS Spam Detector

This project is part of the Module 21 Challenge at Columbia University's AI Bootcamp. The objective is to build and deploy an SMS spam classification model using Python. The project involves data preprocessing, machine learning for classification, and setting up an interactive web-based application using Gradio.

## Project Structure

```
.
├── README.md
├── sms_spam_detector.ipynb
├── Resources
│   └── SMSSpamCollection.csv
└── requirements.txt
```

- `README.md`: This file contains the project overview and instructions.
- `sms_spam_detector.ipynb`: The Jupyter notebook containing the code for loading data, building the model, creating the Gradio app, and testing the predictions.
- `Resources/SMSSpamCollection.csv`: The dataset used for training and testing the model.
- `requirements.txt`: A list of dependencies required to run the project.

## Installation

1. **Clone the Repository**:
    ```bash
    git clone https://github.com/KenStager/sms_spam_detector.git
    cd sms_spam_detector
    ```

2. **Create a Virtual Environment**:
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
    ```

3. **Install Dependencies**:
    ```bash
    pip install -r requirements.txt
    ```

## Running the Project

1. **Open the Jupyter Notebook**:
    ```bash
    jupyter notebook sms_spam_detector.ipynb
    ```

2. **Execute the Cells**:
    - Start by running all cells sequentially. The notebook includes cells for loading the dataset, building the model, and setting up the Gradio app.

3. **Launch the Gradio App**:
    - The last cell in the notebook launches the Gradio app. This will open a web interface where you can input SMS messages and get predictions.

## Usage

- Enter an SMS message in the input textbox.
- Click "Submit".
- The prediction will be displayed, indicating whether the message is classified as "spam" or "not spam".

## Example Predictions

1. "You are a lucky winner of $5000!"
2. "You won 2 free tickets to the Super Bowl."
3. "You won 2 free tickets to the Super Bowl text us to claim your prize."
4. "Thanks for registering. Text 4343 to receive free updates on medicare."

## Detailed Code Explanation

### Data Loading and Handling
```python
# Load the dataset into a DataFrame
sms_text_df = pd.read_csv('Resources/SMSSpamCollection.csv', sep='\t', names=["label", "text_message"])
```
- The dataset is loaded into a Pandas DataFrame. The CSV file is separated by tabs and has two columns: `label` and `text_message`.

### Model Building and Training
```python
def sms_classification(sms_text_df):
    X = sms_text_df['text_message']
    y = sms_text_df['label']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33, random_state=42)
    text_clf = Pipeline([('tfidf', TfidfVectorizer(stop_words='english')), ('clf', LinearSVC())])
    text_clf.fit(X_train, y_train)
    return text_clf

# Call the function
text_clf = sms_classification(sms_text_df)
```
- The function `sms_classification` builds a pipeline with TF-IDF vectorization and LinearSVC for classification. It splits the data into training and testing sets, fits the model, and returns the trained pipeline.

### Prediction Function
```python
def sms_prediction(text):
    prediction = text_clf.predict([text])[0]
    if prediction == 'ham':
        return f'The text message: "{text}", is not spam.'
    else:
        return f'The text message: "{text}", is spam.'
```
- The function `sms_prediction` takes an SMS text as input and returns a message indicating whether the text is classified as spam or not.

### Gradio Interface
```python
def sms_prediction_interface(text):
    return sms_prediction(text)

interface = gr.Interface(
    fn=sms_prediction_interface, 
    inputs=gr.Textbox(lines=2, placeholder="Enter SMS text here...", label="SMS Text"), 
    outputs=gr.Textbox(label="Prediction"),
    title="SMS Spam Detector",
    description="Enter an SMS message to determine if it is spam or not.",
)

interface.launch(share=True)
```
- The Gradio interface is set up with the `sms_prediction_interface` function. It takes input through a textbox and outputs the prediction. The app is launched with `share=True` to provide a public URL.

## Requirements

- pandas
- scikit-learn
- gradio

These can be installed via `pip install -r requirements.txt`.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Columbia University AI Bootcamp for providing the project guidelines and dataset.
- [Gradio](https://www.gradio.app) for the interactive web interface library.
