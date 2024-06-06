from transformers import BertTokenizer, TFBertForSequenceClassification
from deep_translator import GoogleTranslator

def get_tokenizer(model_name):
    return BertTokenizer.from_pretrained(model_name)

def get_model(pretrained_path):
    return TFBertForSequenceClassification.from_pretrained(pretrained_path)

def translate_to_indo(text):
    translator = GoogleTranslator(source='en', target='id')
    translated_text = translator.translate(text)
    return translated_text

def predict_sentiment(text, tokenizer, model, max_length):
    translated_text = translate_to_indo(text)
    tokenized_text = tokenizer(
        text=translated_text,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    input_ids = tokenized_text['input_ids']
    attention_mask = tokenized_text['attention_mask']
    prediction = model.predict([input_ids, attention_mask])
    sentiment = "Positive" if prediction[0][0][1] >= 1 else "Negative"
    return sentiment