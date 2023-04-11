import gradio as gr

def sentence_builder(quantity, animal, place, activity_list, morning):
    return f"""The {quantity} {animal}s went to the {place} where they {" and ".join(activity_list)} until the {"morning" if morning else "night"}"""


demo = gr.Interface(
    sentence_builder,
    [
        gr.Slider(2, 20, value=4),
        gr.Dropdown(["cat", "dog", "bird"]),
        gr.Radio(["park", "zoo", "road"]),
        gr.Dropdown(["ran", "swam", "ate", "slept"], value=["swam", "slept"], multiselect=True),
        gr.Checkbox(label="Is it the morning?"),
    ],
    "text",
    examples=[
        [2, "cat", "park", ["ran", "swam"], True],
        [4, "dog", "zoo", ["ate", "swam"], False],
        [10, "bird", "road", ["ran"], False],
        [8, "cat", "zoo", ["ate"], True],
    ],
)

if __name__ == "__main__":
    demo.launch()