from transformers import (
    TFAutoModelWithLMHead,
    AutoTokenizer,
    pipeline,
    BlenderbotTokenizer,
    BlenderbotSmallTokenizer,
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

#################################################################################################################
# from transformers import AutoModelForMaskedLM, AutoTokenizer                                                  #
#                                                                                                               #
# speech_model = AutoModelForMaskedLM.from_pretrained("facebook/wav2vec2-base-960h")                            #
# tokenizer = AutoTokenizer.from_pretrained("facebook/wav2vec2-base-960h", do_lower_case=True)                  #
# from datasets import load_dataset                                                                             #
# import soundfile as sf                                                                                        #
#                                                                                                               #
# # use "dummy" samples of validation split because `load_dataset("librispeech_asr", "clean")` requires > 50GB  #
# libri_speech_dummy = load_dataset("patrickvonplaten/librispeech_asr_dummy", "clean", split="validation")      #
#                                                                                                               #
# # define function to read in audio file                                                                       #
# def map_to_array(batch):                                                                                      #
#   speech, _ = sf.read(batch["file"])                                                                          #
#   batch["speech"] = speech                                                                                    #
#   return batch                                                                                                #
#                                                                                                               #
# samples = libri_speech_dummy.map(map_to_array)[5:8]                                                           #
#################################################################################################################
