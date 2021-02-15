from transformers import (
    TFAutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
    BlenderbotTokenizer,
    BlenderbotForConditionalGeneration,
    Conversation,
)

import logging

def blenderbot400M(utterance: str):
    from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

    tokenizer = AutoTokenizer.from_pretrained("facebook/blenderbot-400M-distill")
    model = AutoModelForSeq2SeqLM.from_pretrained("facebook/blenderbot-400M-distill")
    inputs = tokenizer([utterance], return_tensors="pt")
    reply_ids = model.generate(**inputs)
    responses = [
        tokenizer.decode(g, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        for g in reply_ids
    ]
    return responses

def compute_sentiment(utterance: str) -> float:
    nlp = pipeline("sentiment-analysis")
    result = nlp(utterance)
    score = result[0]["score"]
    if result[0]["label"] == "NEGATIVE":
        score = score * -1

    logging.info("The score was {}".format(score))
    return score
