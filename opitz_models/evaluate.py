import subprocess
import json

### Loading data
# Validation Set & Generated Questions
evaluation_file_path = '/localdata1/opit_do/critical_question_generation/st_critical_questions/data_splits/sample_copy.json'
generated_questions_path = '/localdata1/opit_do/critical_question_generation/st_critical_questions/output/67450_output.json'
#generated_questions_path = evaluation_file_path


### Run evaluation script
# Define arguments to pass to the evaluate.py script
args = [
    'python3',
    '/localdata1/opit_do/critical_question_generation/st_critical_questions/eval_scripts/evaluation.py',
    '--metric', 'similarity',
    '--input_path', evaluation_file_path, # Path of the test set.
    '--submission_path', generated_questions_path, # Path where the generated questions have been saved.
    '--threshold', '0.7'
]

# Run the script
subprocess.run(args)