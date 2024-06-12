from transformers import BertTokenizer, TFBertForSequenceClassification
from deep_translator import GoogleTranslator

def get_tokenizer(model_name):
    return BertTokenizer.from_pretrained(model_name)

def get_model(pretrained_path):
    return TFBertForSequenceClassification.from_pretrained(pretrained_path)

# dump
def translate_to_indo(text):
    translator = GoogleTranslator(source='en', target='id')
    translated_text = translator.translate(text)
    return translated_text

def predict_results(texts, tokenizer, model, max_length):
    tokenized_texts = tokenizer(
        text=texts,
        add_special_tokens=True,
        max_length=max_length,
        truncation=True,
        padding='max_length',
        return_tensors='tf'
    )
    input_ids = tokenized_texts['input_ids']
    attention_masks = tokenized_texts['attention_mask']
    predictions = model.predict([input_ids, attention_masks], use_multiprocessing=True, workers=4)
    return predictions.logits