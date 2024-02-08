import torch
from transformers import AutoTokenizer, AutoModelWithLMHead

#utilizing Google's T5 API in Python using a large scale transformer

tokenizer = AutoTokenizer.from_pretrained('t5-base')
model = AutoModelWithLMHead.from_pretrained('t5-base', return_dict=True)

sequence = ("Over the years, there’ve been many things that I’ve not done because I fear social disapproval." 
            "For example, the fear of rejection has often held me back from asking girls out. Social anxiety is the fear that actions perceived negatively by others will affect my social standing in some way."
            "No advice will completely dissolve your social anxiety. But I can tell you how I’ve worked through mine. For me, understanding where my social anxiety is coming from helped me overcome it. Our fear comes from the possibility of social disapproval."
            "We’re vigilant against any potential hits to our social status and standing in the ‘tribe’. If we think in terms of evolutionary psychobiology, fear serves to ensure our survival. There’s the sort of fear you’d feel when a lion is chasing you and your life is in active danger."
            "But we’re also highly attuned to the judgment of other people in our tribe. As prehistoric men, we lived in hunter-gatherer tribes. If you were ostracised from your tribe, it’d potentially be the end of you. You’d have no one to rely on and would have to go out into the wilderness on your own. It’d be very likely that you’d get eaten by that lion chasing you around."
            "You might even just die of starvation. Being part of a tribe was essential for our survival. So our amygdala – the part of the brain that experiences and processes fear – became very attuned to social threats. The risk of ostracization is processed as a danger to our lives. That’s why we evolved to fear other people’s judgement and rejection." 
            "I’ve been able to shift my perception by recognizing two things: Social anxiety is normal No one really cares what I get up to")
inputs = tokenizer.encode("summarize: " + sequence, return_tensors='pt', max_length=512, truncation=True)
outputs = model.generate(inputs, max_length=200, min_length=80, length_penalty=5., num_beams=2)
summary = tokenizer.decode(outputs[0])
print(summary)
