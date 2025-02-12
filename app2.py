# E-commerce support chatbot
import gradio as gr
from langchain_core.messages import HumanMessage
from e_commerce import graph

class ChatInterface:
    def __init__(self):
        self.config = {"configurable": {"thread_id": "1"}}
        self.chat_history = []
    
    def chat(self, message, history):
        """Process chat messages and maintain history."""
        # Return if empty message
        if not message:
            return "", history
        
        try:
            # Process the message through the chatbot
            final_state = graph.invoke(
                {"messages": [HumanMessage(content=message)]},
                config=self.config
            )
            
            # Get bot's response
            bot_response = final_state["messages"][-1].content
            
            # Add to history as tuples (user_message, bot_message)
            history.append((message, bot_response))
            
            return "", history
        except Exception as e:
            error_message = f"An error occurred: {str(e)}"
            history.append((message, error_message))
            return "", history

    def create_interface(self):
        """Create and configure the Gradio interface."""
        # Define chat interface
        chat_interface = gr.Blocks(theme=gr.themes.Soft())
        
        with chat_interface:
            gr.Markdown("# E-Commerce Support Chatbot")
            gr.Markdown("Welcome to our E-commerce Support! How can I help you today?")
            
            chatbot = gr.Chatbot(
                height=400,
                show_label=False,
                container=True,
            )
            
            with gr.Row():
                msg = gr.Textbox(
                    scale=4,
                    show_label=False,
                    placeholder="Type your message here...",
                    container=False
                )
                submit = gr.Button("Send", scale=1)
            
            # Clear button
            clear = gr.Button("Clear Chat")
            
            # Set up event handlers
            submit_click = submit.click(
                self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            msg_submit = msg.submit(
                self.chat,
                inputs=[msg, chatbot],
                outputs=[msg, chatbot]
            )
            
            clear.click(lambda: None, None, chatbot, queue=False)
            
        return chat_interface

def main():
    # Create chat interface instance
    chat_app = ChatInterface()
    
    # Launch the interface
    chat_app.create_interface().launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True
    )

if __name__ == "__main__":
    main()