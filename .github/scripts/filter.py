import os
import json

full_matrix_string = os.environ["MAT"]
full_matrix = json.loads(full_matrix_string)

new_matrix_entries = []

for entry in full_matrix['include']:
    if entry['gpu_arch_version'] != "12.1":
        new_matrix_entries.append(entry)

new_matrix = {'include': new_matrix_entries}
print(json.dumps(new_matrix))
