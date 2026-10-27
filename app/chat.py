# app/chat.py

from app.api_client import GroqClient

class ChatManager:
    """Class to manage chat interactions."""

    def __init__(self):
        self.client = GroqClient()  # Initialize the Groq client
        self.conversation_history = []  # Store conversation history

    def add_message(self, role, content):
        """Add a message to the conversation history."""
        self.conversation_history.append({"role": role, "content": content})

    def get_response(self, user_message, context_chunks=None):
        """
        Get a concise or detailed response based on user input.

        :param user_message: The message input from the user.
        :param context_chunks: Optional list of retrieved chunks (dicts with
            'source' and 'text') to ground the answer in.
        :return: AI response as a string.
        """
        # Add user's message to history
        self.add_message("user", user_message)

        # Prepare messages for the API call based on user intent
        if any(keyword in user_message.lower() for keyword in ["what is", "define"]):
            prompt = f"Provide a concise definition of: {user_message}"
        
        elif "explain" in user_message.lower() or "code" in user_message.lower():
            prompt = f"Explain clearly with examples: {user_message}"
        
        else:
            prompt = f"Respond concisely to: {user_message}"

        if context_chunks:
            context = "\n\n".join(
                f"[{i}] (source: {c['source']})\n{c['text']}"
                for i, c in enumerate(context_chunks, start=1)
            )
            prompt = (
                "Use the following context excerpts to answer the question. "
                "If the context is not relevant, answer from general knowledge and say so.\n\n"
                f"Context:\n{context}\n\n"
                f"Question instructions: {prompt}"
            )

        # Construct the full messages payload with system instructions, rolling chat memory, and the new query
        messages_to_send = []
        
        # 1. System Prompt (overall persona and guidelines)
        messages_to_send.append({
            "role": "system",
            "content": "You are IntellectBridge, a highly professional corporate assistant. Use the provided context to answer questions."
        })
        
        # 2. Rolling History (exclude latest raw message added on line 26, keep last 6 messages/3 turns)
        history_limit = 6
        recent_history = self.conversation_history[:-1][-history_limit:]
        messages_to_send.extend(recent_history)
        
        # 3. Grounded Query (latest user message, enriched with fresh RAG search context)
        messages_to_send.append({
            "role": "user",
            "content": prompt
        })

        # Get AI response using full multi-turn context
        ai_response = self.client.get_response(messages_to_send)

        # Add AI's response to history
        self.add_message("assistant", ai_response)

        return ai_response