import json

# Load the JSON data
with open('/home/kyueran/caption-generation/BLIP/output/analyse_merlion_pre_distil_0/result/results.json', 'r') as file:
    data = json.load(file)

# Initialize counters
correct_with_merlion = 0
correct_without_merlion = 0

# Check the first 400 captions for the presence of "merlion"
for i in range(50):
    if "merlion" in data[i]["caption"]:
        correct_with_merlion += 1

# Check the next 400 captions for the absence of "merlion"
for i in range(50, 100):
    if "merlion" not in data[i]["caption"]:
        correct_without_merlion += 1

# Calculate accuracy
accuracy_with_merlion = correct_with_merlion / 50
accuracy_without_merlion = correct_without_merlion / 50
overall_accuracy = (correct_with_merlion + correct_without_merlion) / 100

print(f"Accuracy for captions with 'merlion': {accuracy_with_merlion * 100:.2f}%")
print(f"Accuracy for captions without 'merlion': {accuracy_without_merlion * 100:.2f}%")
print(f"Overall accuracy: {overall_accuracy * 100:.2f}%")
