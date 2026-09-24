
import gradio as gr
from app import demo as app
import os

_docs = {'ClickAudio': {'description': 'Creates an audio component that can be used to upload/record audio (as an input) or display audio (as an output).', 'members': {'__init__': {'value': {'type': 'str | Path | tuple[int, np.ndarray] | Callable | None', 'default': 'value = None', 'description': 'A path, URL, or [sample_rate, numpy array] tuple (sample rate in Hz, audio data as a float or int numpy array) for the default value that ClickAudio component is going to take. If a function is provided, the function will be called each time the app loads to set the initial value of this component.'}, 'sources': {'type': "list[Literal['upload', 'microphone']] | Literal['upload', 'microphone'] | None", 'default': 'value = None', 'description': 'A list of sources permitted for audio. "upload" creates a box where user can drop an audio file, "microphone" creates a microphone input. The first element in the list will be used as the default source. If None, defaults to ["upload", "microphone"], or ["microphone"] if `streaming` is True.'}, 'type': {'type': "Literal['numpy', 'filepath']", 'default': 'value = "numpy"', 'description': 'The format the audio file is converted to before being passed into the prediction function. "numpy" converts the audio to a tuple consisting of: (int sample rate, numpy.array for the data), "filepath" passes a str path to a temporary file containing the audio.'}, 'label': {'type': 'str | I18nData | None', 'default': 'value = None', 'description': 'the label for this component. Appears above the component and is also used as the header if there are a table of examples for this component. If None and used in a `gr.Interface`, the label will be the name of the parameter this component is assigned to.'}, 'every': {'type': 'Timer | float | None', 'default': 'value = None', 'description': 'Continuously calls `value` to recalculate it if `value` is a function (has no effect otherwise). Can provide a Timer whose tick resets `value`, or a float that provides the regular interval for the reset Timer.'}, 'inputs': {'type': 'Component | Sequence[Component] | set[Component] | None', 'default': 'value = None', 'description': 'Components that are used as inputs to calculate `value` if `value` is a function (has no effect otherwise). `value` is recalculated any time the inputs change.'}, 'show_label': {'type': 'bool | None', 'default': 'value = None', 'description': 'if True, will display label.'}, 'container': {'type': 'bool', 'default': 'value = True', 'description': 'If True, will place the component in a container - providing some extra padding around the border.'}, 'scale': {'type': 'int | None', 'default': 'value = None', 'description': 'Relative width compared to adjacent Components in a Row. For example, if Component A has scale=2, and Component B has scale=1, A will be twice as wide as B. Should be an integer.'}, 'min_width': {'type': 'int', 'default': 'value = 160', 'description': 'Minimum pixel width, will wrap if not sufficient screen space to satisfy this value. If a certain scale value results in this Component being narrower than min_width, the min_width parameter will be respected first.'}, 'interactive': {'type': 'bool | None', 'default': 'value = None', 'description': 'If True, will allow users to upload and edit an audio file. If False, can only be used to play audio. If not provided, this is inferred based on whether the component is used as an input or output.'}, 'visible': {'type': "bool | Literal['hidden']", 'default': 'value = True', 'description': 'If False, component will be hidden. If "hidden", component will be visually hidden and not take up space in the layout but still exist in the DOM If "hidden", component will be visually hidden and not take up space in the layout but still exist in the DOM.'}, 'streaming': {'type': 'bool', 'default': 'value = False', 'description': 'If set to True when used in a `live` interface as an input, will automatically stream webcam feed. When used set as an output, takes audio chunks yield from the backend and combines them into one streaming audio output.'}, 'elem_id': {'type': 'str | None', 'default': 'value = None', 'description': 'An optional string that is assigned as the id of this component in the HTML DOM. Can be used for targeting CSS styles.'}, 'elem_classes': {'type': 'list[str] | str | None', 'default': 'value = None', 'description': 'An optional list of strings that are assigned as the classes of this component in the HTML DOM. Can be used for targeting CSS styles.'}, 'render': {'type': 'bool', 'default': 'value = True', 'description': 'if False, component will not be rendered in the Blocks context. Should be used if the intention is to assign event listeners now but render the component later.'}, 'key': {'type': 'int | str | tuple[int | str, ...] | None', 'default': 'value = None', 'description': "in a gr.render, Components with the same key across re-renders are treated as the same component, not a new component. Properties set in 'preserved_by_key' are not reset across a re-render."}, 'preserved_by_key': {'type': 'list[str] | str | None', 'default': 'value = "value"', 'description': "A list of parameters from this component's constructor. Inside a gr.render() function, if a component is re-rendered with the same key, these (and only these) parameters will be preserved in the UI (if they have been changed by the user or an event listener) instead of re-rendered based on the values provided during constructor."}, 'format': {'type': "Literal['wav', 'mp3'] | None", 'default': 'value = None', 'description': 'the file extension with which to save audio files. Either \'wav\' or \'mp3\'. wav files are lossless but will tend to be larger files. mp3 files tend to be smaller. This parameter applies both when this component is used as an input (and `type` is "filepath") to determine which file format to convert user-provided audio to, and when this component is used as an output to determine the format of audio returned to the user. If None, no file format conversion is done and the audio is kept as is. In the case where output audio is returned from the prediction function as numpy array and no `format` is provided, it will be returned as a "wav" file.'}, 'autoplay': {'type': 'bool', 'default': 'value = False', 'description': 'Whether to automatically play the audio when the component is used as an output. Note: browsers will not autoplay audio files if the user has not interacted with the page yet.'}, 'editable': {'type': 'bool', 'default': 'value = True', 'description': 'If True, allows users to manipulate the audio file if the component is interactive. Defaults to True.'}, 'buttons': {'type': "list[Literal['download', 'share'] | Button] | None", 'default': 'value = None', 'description': 'A list of buttons to show in the top right corner of the component. Valid options are "download", "share", or a gr.Button() instance. The "download" button allows the user to save the audio to their device. The "share" button allows the user to share the audio via Hugging Face Spaces Discussions. Custom gr.Button() instances will appear in the toolbar with their configured icon and/or label, and clicking them will trigger any .click() events registered on the button. By default, only the "download" and "share" buttons are shown.'}, 'waveform_options': {'type': 'WaveformOptions | dict | None', 'default': 'value = None', 'description': 'A dictionary of options for the waveform display. Options include: waveform_color (str), waveform_progress_color (str), skip_length (int), trim_region_color (str). Default is None, which uses the default values for these options. [See `gr.WaveformOptions` docs](#waveform-options).'}, 'loop': {'type': 'bool', 'default': 'value = False', 'description': 'If True, the audio will loop when it reaches the end and continue playing from the beginning.'}, 'recording': {'type': 'bool', 'default': 'value = False', 'description': 'If True, the audio component will be set to record audio from the microphone if the source is set to "microphone". Defaults to False.'}, 'subtitles': {'type': 'str | Path | list[dict[str, Any]] | None', 'default': 'value = None', 'description': 'A subtitle file (srt, vtt, or json) for the audio, or a list of subtitle dictionaries in the format [{"text": str, "timestamp": [start, end]}] where timestamps are in seconds. JSON files should contain an array of subtitle objects.'}, 'playback_position': {'type': 'float', 'default': 'value = 0', 'description': 'The starting playback position in seconds. This value is also updated as the audio plays, reflecting the current playback position.'}}, 'postprocess': {'value': {'type': 'str| pathlib.Path| bytes| tuple[int,numpy.ndarray]| None', 'description': 'Expects audio data in any of these formats:'}}, 'preprocess': {'return': {'type': 'str| tuple[int,numpy.ndarray]| None', 'description': 'Passes audio as one of these formats (depending on `type`):'}, 'value': None}}, 'events': {'stream': {'type': None, 'default': None, 'description': 'This listener is triggered when the user streams the ClickAudio.'}, 'change': {'type': None, 'default': None, 'description': 'Triggered when the value of the ClickAudio changes either because of user input (e.g. a user types in a textbox) OR because of a function update (e.g. an image receives a value from the output of an event trigger). See `.input()` for a listener that is only triggered by user input.'}, 'clear': {'type': None, 'default': None, 'description': 'This listener is triggered when the user clears the ClickAudio using the clear button for the component.'}, 'play': {'type': None, 'default': None, 'description': 'This listener is triggered when the user plays the media in the ClickAudio.'}, 'pause': {'type': None, 'default': None, 'description': 'This listener is triggered when the media in the ClickAudio stops for any reason.'}, 'stop': {'type': None, 'default': None, 'description': 'This listener is triggered when the user reaches the end of the media playing in the ClickAudio.'}, 'start_recording': {'type': None, 'default': None, 'description': 'This listener is triggered when the user starts recording with the ClickAudio.'}, 'pause_recording': {'type': None, 'default': None, 'description': 'This listener is triggered when the user pauses recording with the ClickAudio.'}, 'stop_recording': {'type': None, 'default': None, 'description': 'This listener is triggered when the user stops recording with the ClickAudio.'}, 'upload': {'type': None, 'default': None, 'description': 'This listener is triggered when the user uploads a file into the ClickAudio.'}, 'input': {'type': None, 'default': None, 'description': 'This listener is triggered when the user changes the value of the ClickAudio.'}, 'seek': {'type': None, 'default': None, 'description': 'This listener is triggered when the user clicks on the waveform to seek to a new playback position. Add `evt: gr.EventData` to the listener function and read `evt.time` for the clicked time, in seconds.'}}}, '__meta__': {'additional_interfaces': {}, 'user_fn_refs': {'ClickAudio': []}}}

abs_path = os.path.join(os.path.dirname(__file__), "css.css")

with gr.Blocks(
    css=abs_path,
    theme=gr.themes.Default(
        font_mono=[
            gr.themes.GoogleFont("Inconsolata"),
            "monospace",
        ],
    ),
) as demo:
    gr.Markdown(
"""
# `gradio_clickaudio`

<div style="display: flex; gap: 7px;">
<img alt="Static Badge" src="https://img.shields.io/badge/version%20-%200.0.1%20-%20orange">  
</div>

Python library for easily interacting with trained machine learning models
""", elem_classes=["md-custom"], header_links=True)
    app.render()
    gr.Markdown(
"""
## Installation

```bash
pip install gradio_clickaudio
```

## Usage

```python

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

```
""", elem_classes=["md-custom"], header_links=True)


    gr.Markdown("""
## `ClickAudio`

### Initialization
""", elem_classes=["md-custom"], header_links=True)

    gr.ParamViewer(value=_docs["ClickAudio"]["members"]["__init__"], linkify=[])


    gr.Markdown("### Events")
    gr.ParamViewer(value=_docs["ClickAudio"]["events"], linkify=['Event'])




    gr.Markdown("""

### User function

The impact on the users predict function varies depending on whether the component is used as an input or output for an event (or both).

- When used as an Input, the component only impacts the input signature of the user function.
- When used as an output, the component only impacts the return signature of the user function.

The code snippet below is accurate in cases where the component is used as both an input and an output.

- **As input:** Is passed, passes audio as one of these formats (depending on `type`):.
- **As output:** Should return, expects audio data in any of these formats:.

 ```python
def predict(
    value: str| tuple[int,numpy.ndarray]| None
) -> str| pathlib.Path| bytes| tuple[int,numpy.ndarray]| None:
    return value
```
""", elem_classes=["md-custom", "ClickAudio-user-fn"], header_links=True)




    demo.load(None, js=r"""function() {
    const refs = {};
    const user_fn_refs = {
          ClickAudio: [], };
    requestAnimationFrame(() => {

        Object.entries(user_fn_refs).forEach(([key, refs]) => {
            if (refs.length > 0) {
                const el = document.querySelector(`.${key}-user-fn`);
                if (!el) return;
                refs.forEach(ref => {
                    el.innerHTML = el.innerHTML.replace(
                        new RegExp("\\b"+ref+"\\b", "g"),
                        `<a href="#h-${ref.toLowerCase()}">${ref}</a>`
                    );
                })
            }
        })

        Object.entries(refs).forEach(([key, refs]) => {
            if (refs.length > 0) {
                const el = document.querySelector(`.${key}`);
                if (!el) return;
                refs.forEach(ref => {
                    el.innerHTML = el.innerHTML.replace(
                        new RegExp("\\b"+ref+"\\b", "g"),
                        `<a href="#h-${ref.toLowerCase()}">${ref}</a>`
                    );
                })
            }
        })
    })
}

""")

demo.launch()
