
import gradio as gr
from gradio_clickaudio import ClickAudio


example = ClickAudio().example_value()

demo = gr.Interface(
    lambda x:x,
    ClickAudio(),  # interactive version of your component
    ClickAudio(),  # static version of your component
    # examples=[[example]],  # uncomment this line to view the "example version" of your component
)


if __name__ == "__main__":
    demo.launch()
