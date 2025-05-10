import json
import os

# Load the current notebook
with open('notebooks/semantic_search_project.ipynb', 'r') as f:
    notebook = json.load(f)

# Find the evaluate_retrieval function cell
for cell in notebook['cells']:
    if cell['cell_type'] == 'code' and 'evaluate_retrieval' in ''.join(cell['source']):
        source = cell['source']
        
        # Add the type conversion line (convert numpy.int64 to Python int)
        for i, line in enumerate(source):
            if 'for idx in sample_indices:' in line:
                # Insert the type conversion after this line
                source.insert(i+1, "        # Convert numpy.int64 to regular Python int to avoid type issues\n")
                source.insert(i+2, "        idx = int(idx)\n")
                break
        
        # Update the cell
        cell['source'] = source
        break

# Save the fixed notebook
with open('notebooks/semantic_search_project.ipynb', 'w') as f:
    json.dump(notebook, f, indent=1)

print("Notebook fixed successfully!") 