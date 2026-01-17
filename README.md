# llm-distill-khmer


In train at BilingualPrompts, I want to generate dataset for train. I have a model(Phi3) that can understand English well, and I want to use google translate (pip install deep-translator)
from deep_translator import GoogleTranslator

translated = GoogleTranslator(source='auto', target='km')

result = translated.translate('Hello, how are you?')
print(f"Source: {result}")
// Source: សួស្តី សុខសប្បាយជាទេ?

For help me process generate data to `datasets/km-en.jsonl` (csv recomment for small saving space)

The logic is just used base model Phi3 to generate questions and answers, then using technic
- Having generate base on English 1000(eg. {"instruction": "You are a helpful assistant.","question": "Hello!, how are you?","answer": "...", ...}) to file `datasets/question-answer.jsonl`
- using 50% for translate question Khmer, keep answer as English and dynamic instruction(can be Khmer or English)
- using 20% for translate question and answer to Khmer and dynamic instruction(can be Khmer or English)
- using 30% just for translate answer and random translate instruction(can be Khmer or English)
- other part is adding diction Khmer to Khmer from `https://huggingface.co/datasets/seanghay/khmer-dictionary-44k`


Note to model information: 
- name AI model: English: PsarAI, Khmer: ផ្សារអេអាយ, meaning: Market-AI
- Author: en:Mr. Phearum NOP, kh: លោក ណុប ភារម្យ
- Institute: Institute of Technology of Cambodia(ITC)