import json
import os

# Path to the notebook
backup_path = 'notebooks/semantic_search_news_media.ipynb.bak'
output_path = 'notebooks/semantic_search_news_media.ipynb'

try:
    # Read the file as text
    with open(backup_path, 'r') as f:
        content = f.read()
    
    # Find the end of the valid JSON
    end_pos = content.find('}\n## Live Demo')
    
    if end_pos > 0:
        valid_json = content[:end_pos+1]  # Include the closing bracket
        
        try:
            # Parse the notebook
            notebook = json.loads(valid_json)
            
            # Create a new markdown cell
            live_demo_cell = {
                "cell_type": "markdown",
                "metadata": {},
                "source": [
                    "## Live Demo\n",
                    "\n",
                    "You can access a live demo of this semantic search system at:\n",
                    "[https://9flabixjy2bbz7fvhfyzgg.streamlit.app/](https://9flabixjy2bbz7fvhfyzgg.streamlit.app/)\n",
                    "\n",
                    "The demo allows you to try out semantic search queries on the AG News dataset and explore the capabilities of the system without needing to run the code locally."
                ]
            }
            
            # Add the new cell to the notebook
            notebook['cells'].append(live_demo_cell)
            
            # Write the updated notebook
            with open(output_path, 'w') as f:
                json.dump(notebook, f, indent=1)
            
            print("Successfully fixed the notebook!")
        except json.JSONDecodeError as e:
            print(f"JSON parsing error: {e}")
    else:
        print("Could not find the boundary between valid JSON and appended text")
except Exception as e:
    print(f"Error: {e}")
    print("Notebook might still be corrupted.") 