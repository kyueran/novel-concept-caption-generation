import torch
from PIL import Image
import requests
from transformers import BitsAndBytesConfig, pipeline
from sentence_transformers import SentenceTransformer, util
import spacy
from nltk.corpus import wordnet as wn

# Load spaCy model
nlp = spacy.load("en_core_web_sm")

# Define the image URL and load the image
image_url = "https://llava-vl.github.io/static/images/view.jpg"
image = Image.open(requests.get(image_url, stream=True).raw)

# Configure quantization for the model
quantization_config = BitsAndBytesConfig(
    load_in_4bit=True,
    bnb_4bit_compute_dtype=torch.float16
)

# Initialize the pipeline for image-to-text
image_model_id = "llava-hf/llava-v1.6-mistral-7b-hf"
pipe_image_to_text = pipeline("image-to-text", model=image_model_id, model_kwargs={"quantization_config": quantization_config})

summarizer = pipeline("summarization", model="t5-small", tokenizer="t5-small")

# Function to convert image to text using the Llava model
def convert_image_to_text(image, pipe):
    prompt = "USER: <image>\nPlease describe this image.\nASSISTANT:"
    outputs = pipe(image, prompt=prompt, generate_kwargs={"max_new_tokens": 200})
    response = outputs[0]["generated_text"]
    # Extract text after "ASSISTANT:"
    assistant_index = response.find("ASSISTANT:")
    if assistant_index != -1:
        response = response[assistant_index + len("ASSISTANT:"):].strip()
    return response

# Extract objects and features using spaCy
def extract_objects_features(text):
    doc = nlp(text)
    objects_features = set()
    for token in doc:
        if token.pos_ in {'NOUN'}:
            objects_features.add(token.lemma_.lower())
    return objects_features

# Create a dictionary of synonyms
def create_synonym_dict(objects_features):
    synonym_dict = {}
    for obj in objects_features:
        synonyms = set([obj])
        for syn in wn.synsets(obj):
            for lemma in syn.lemmas():
                synonyms.add(lemma.name().lower())
        synonym_dict[obj] = synonyms
    return synonym_dict

# Match objects/features with synonyms
def match_with_synonyms(objects_features1, synonym_dict2):
    matched_objects = set()
    for obj1 in objects_features1:
        for obj2, synonyms in synonym_dict2.items():
            if obj1 in synonyms:
                matched_objects.add(obj1)
                break
    return matched_objects

# Calculate precision, recall, and F1 Score
def calculate_precision_recall_f1(set1, set2, matched_objects):
    true_positives = len(matched_objects)
    possible_positives = len(set1)
    predicted_positives = len(set2)

    precision = true_positives / predicted_positives if predicted_positives > 0 else 0
    recall = true_positives / possible_positives if possible_positives > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0

    return precision, recall, f1

def summarize(caption):
    summary = summarizer(caption, max_length=50, do_sample=False)
    summarized = summary[0]['summary_text']
    return summarized

generated_caption = convert_image_to_text(image, pipe_image_to_text)
student_caption = "This image shows a dock to a lake with mountains and trees in the background."
teacher_caption = summarize(generated_caption)
print(teacher_caption)

# Extract objects and features from both captions
objects_features_generated = extract_objects_features(teacher_caption)
objects_features_existing = extract_objects_features(student_caption)

# Create synonym dictionaries
synonym_dict_generated = create_synonym_dict(objects_features_generated)
synonym_dict_existing = create_synonym_dict(objects_features_existing)
print("Generated Features", objects_features_generated)
print("Existing Features", objects_features_existing)
print()

# Match objects/features with synonyms
matched_objects_from_existing = match_with_synonyms(objects_features_existing, synonym_dict_generated)
matched_objects_from_generated = match_with_synonyms(objects_features_generated, synonym_dict_existing)
print()
print("MATCHED EXISTING", matched_objects_from_existing)
print()
print("MATHCED GENERATED", matched_objects_from_generated)

# Calculate precision, recall, and F1 Score for objects and features
precision, recall, f1 = calculate_precision_recall_f1(objects_features_existing, objects_features_generated, matched_objects_from_existing)
print(f"Precision: {precision}, Recall: {recall}, F1 Score for Objects and Features: {f1}")

# Load the paraphrase-MiniLM-L6-v2 model
model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Encode the captions
embeddings_teacher = model.encode(teacher_caption, convert_to_tensor=True)
embeddings_student = model.encode(student_caption, convert_to_tensor=True)
embeddings_unsummarized = model.encode(generated_caption, convert_to_tensor=True)
# Calculate cosine similarity
similarity_score1 = util.pytorch_cos_sim(embeddings_student, embeddings_teacher).item()
similarity_score2 = util.pytorch_cos_sim(embeddings_teacher, embeddings_student).item()
similarity_score3 = util.pytorch_cos_sim(embeddings_student, embeddings_unsummarized).item()

print(f"Semantic Similarity Score: {similarity_score1}, {similarity_score2}, {similarity_score3}")

# Combine F1 score and Semantic Similarity score
final_score = f1 + similarity_score1
print(f"Final Similarity Score: {final_score}")
