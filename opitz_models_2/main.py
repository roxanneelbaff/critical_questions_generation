import json
import random
from QuestionScorer import QuestionScorer
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from collections import defaultdict

LLM = "llama3.2:latest" # "llama3.2:3b-instruct-q8_0"

# Load arguments
filename = "sample.json"
path = f'/localdata1/opit_do/critical_question_generation/st_critical_questions/data_splits/{filename}'
with open(path, 'r') as file:
    arguments_data = json.load(file)
output_data = arguments_data

# Score questions
statistics = {}
scorer = QuestionScorer(LLM)
argument_count = 1
total_questions = 0
for key, value in output_data.items():
    print(f"Argument {argument_count}")
    argument = value["intervention"]
    cqs = value["cqs"]
    question_count = 1
    for cq in cqs:
        question = cq["cq"]
        label = cq["label"]
        print(f" > Question {question_count}/{len(cqs)} - Score: ", end="")
        score = scorer.score_question(argument, question)
        print(f"{score}, Actual Label: {label}")
        cq["score"] = score
        if label not in statistics:
            statistics[label] = []
        statistics[label].append(score)
        question_count += 1
        total_questions += 1
    argument_count += 1

# Save scored dataset to file
suffix = random.randint(1000, 9999)
filename = f"sample_scored_{suffix}"
output_path = f'/localdata1/opit_do/critical_question_generation/st_critical_questions/output/new/{filename}.json'
with open(output_path, 'w') as file:
    json.dump(output_data, file, indent=4)
    print(f"Dumped output to {output_path}")

# Visualize scores
# Define the range of possible scores (0 to 10)
all_scores = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]  # This will ensure the y-axis shows all integers from 0 to 10

# Find unique categories (x-axis)
labels = list(statistics.keys())

# Create a frequency matrix to store counts of each score per label
frequency_matrix = np.zeros((len(all_scores), len(labels)))

# Fill the matrix with frequencies
for j, label in enumerate(labels):
    for value in statistics[label]:
        if value in all_scores:
            i = all_scores.index(value)  # Find the row corresponding to the score
            frequency_matrix[i, j] += 1  # Increment the count for this label and score

# Create the heatmap using seaborn
plt.figure(figsize=(10, 6))
sns.heatmap(frequency_matrix, annot=True, xticklabels=labels, yticklabels=all_scores, cmap='Blues')

# Add labels and title
plt.xlabel('True Labels')
plt.ylabel('LLM-assigned Scores')
plt.title(f'Frequency of Scores Assigned to Labels\n')
plt.suptitle(f'(First {total_questions} Questions of {filename}), {LLM}', fontsize=8, x=0.5, y=0.9)

# Flip the y-axis so that 0 is at the bottom and 10 is at the top
plt.gca().invert_yaxis()

# Show the plot
plt.show()

plt.savefig(f"/localdata1/opit_do/critical_question_generation/st_critical_questions/output/new/{filename}.png", format='png', dpi=300)
print(f"Dumped plot to /localdata1/opit_do/critical_question_generation/st_critical_questions/output/new/{filename}.png")