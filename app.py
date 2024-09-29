import gradio as gr
from prompt_parser import Parse_Prompt
from scoreboard import Score
import warnings

warnings.filterwarnings("ignore")

arena = Parse_Prompt()
score = Score()

with gr.Blocks(fill_height = True) as app:
    with gr.Tab("🪖 Battle Field"):
        gr.Markdown('''## ⚔️ LLM: Large Language Mayhem
                    - Voting should be fair and based on the performance of the models.
                    - No cheating or manipulating the outcomes.
                    - Press 🎲 Random to change the models.
                    - Everything else except the Random button will only clear the screen, model being the same.
                    - Have fun and enjoy the language mayhem!
                    - Warrior names will be visible after your first query (after random also you will have to give a query to see changes)
                    - Don't See Warrior names before voting
                    ''')
        with gr.Row():
            with gr.Accordion("🥷 Warriors", open = False):
                gr.Dataframe([[model] for model in arena.models], col_count = 1, headers = ["🥷"])
        with gr.Group():
            with gr.Row():
                with gr.Column():
                    chatbox1 = gr.Chatbot(label = "Warrior A", show_copy_button = True)
                with gr.Column():
                    chatbox2 = gr.Chatbot(label = "Warrior B", show_copy_button = True)
            textbox = gr.Textbox(show_label = False, placeholder = "👉 Enter your prompt")
        with gr.Row():
            with gr.Accordion("🥷 Current Warriors",open = False):
                with gr.Row():
                    war1= gr.Textbox(arena.model1, interactive= False, show_label=False)
                    war2 = gr.Textbox(arena.model2, interactive= False, show_label= False)
        with gr.Row():
            with gr.Accordion("👆 Vote", open = False):
                with gr.Row():
                    vote_a = gr.ClearButton([textbox, chatbox1, chatbox2], value = "👈 Warrior A Wins")
                    vote_b = gr.ClearButton([textbox, chatbox1, chatbox2], value = "👉 Warrior B Wins")
                    vote_tie = gr.ClearButton([textbox, chatbox1, chatbox2], value = "🤝 Both Won")

        submit_button = gr.Button("Submit")
        with gr.Row():
            new_round = gr.ClearButton( [textbox, chatbox1, chatbox2], value = "🎲New Round🎲")
            clear = gr.ClearButton([textbox, chatbox1, chatbox2], value = "🧹 Clear")
        with gr.Row():
            with gr.Accordion("🔩 Parameters", open = False):
                temp_slider = gr.Slider(0,1,value = 0.7, step=0.1, label = "Temprature")

        textbox.submit(
            fn = arena.gen_output,
            inputs = [temp_slider, textbox],
            outputs = [chatbox1, chatbox2]
        )
        textbox.submit(
            fn = arena.current_model2,
            outputs = war2
        )
        textbox.submit(
            fn = arena.current_model1,
            outputs = war1
        )
        submit_button.click(
            fn = arena.gen_output,
            inputs = [temp_slider, textbox],
            outputs = [chatbox1, chatbox2]
        )
        submit_button.click(
            fn = arena.current_model1,
            outputs = war1
        )
        submit_button.click(
            fn = arena.current_model2,
            outputs = war2
        )
        vote_a.click(
            fn=lambda: score.update(arena.model1, score.df)
        )
        vote_b.click(
            fn = lambda: score.update(arena.model2, score.df)
        )
        vote_tie.click(
            fn = arena.change_models
        )
        new_round.click(
            fn = arena.change_models
        )
        clear.click(
            fn = arena.clear_history
        )

    with gr.Tab("💯 Score Board") as data_tab:
        gr.Markdown('''## ⚔️ LLM: Large Language Mayhem
                    - Voting should be fair and based on the performance of the models.
                    - No cheating or manipulating the outcomes.
                    - Click on Generate button to Update the 💯 Scoreboard.
                    ''')
        gr.Interface(
            fn = score.df_show,
            inputs = None,
            outputs=gr.Dataframe(type="pandas", label="Scoreboard", headers = ["","",""]),
            live = True,
            allow_flagging = "never",
            clear_btn = None

        )
        

app.launch()