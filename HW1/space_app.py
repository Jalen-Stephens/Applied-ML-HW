import gradio as gr
from pathlib import Path

# Read the markdown file
def load_notes():
    notes_path = Path(__file__).parent / "non_ai_notes.md"
    with open(notes_path, "r", encoding="utf-8") as f:
        return f.read()

# Create the Gradio interface
with gr.Blocks(title="Applied ML HW1 - Non-AI Notes") as demo:
    gr.Markdown("# Applied ML HW1 - Non-AI Notes")
    gr.Markdown("This document contains my thought process and approach to solving the homework problems.")
    
    notes_content = gr.Markdown(load_notes())
    
    refresh_btn = gr.Button("Refresh Notes")
    refresh_btn.click(fn=load_notes, outputs=notes_content)

if __name__ == "__main__":
    demo.launch()
