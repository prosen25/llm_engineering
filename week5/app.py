from dotenv import load_dotenv
import gradio as gr

from implementation.answer import answer_question

load_dotenv(override=True)

def main():
    gr.ChatInterface(fn=answer_question, type="messages").launch(inbrowser=True)

if __name__ == "__main__":
    main()