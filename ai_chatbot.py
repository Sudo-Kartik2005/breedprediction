"""
AI Chatbot for Cattle Breed Information
Uses a knowledge base and pattern matching to answer questions about cattle breeds
"""

import json
import re
from pathlib import Path
from typing import List, Dict, Tuple

class CattleBreedChatbot:
    def __init__(self, class_names_path: Path = Path("class_names.json")):
        """Initialize the chatbot with breed knowledge."""
        self.class_names = self._load_class_names(class_names_path)
        self.breed_info = self._initialize_breed_knowledge()
        self.greetings = ["hi", "hello", "hey", "greetings", "good morning", "good afternoon"]
        self.help_keywords = ["help", "what can you do", "features", "capabilities"]
        
    def _load_class_names(self, path: Path) -> List[str]:
        """Load class names from JSON file."""
        try:
            with open(path, 'r') as f:
                return json.load(f)
        except:
            return []
    
    def _initialize_breed_knowledge(self) -> Dict[str, Dict]:
        """Initialize knowledge base about cattle breeds."""
        return {
            "Alambadi": {
                "origin": "Tamil Nadu, India",
                "type": "Draft",
                "characteristics": "Strong, hardy, used for agricultural work",
                "color": "Grey or white"
            },
            "Gir": {
                "origin": "Gujarat, India",
                "type": "Dairy",
                "characteristics": "High milk yield, distinctive hump, drooping ears",
                "color": "Reddish-brown with white spots"
            },
            "Sahiwal": {
                "origin": "Punjab, Pakistan/India",
                "type": "Dairy",
                "characteristics": "Excellent milk production, heat tolerant",
                "color": "Reddish-brown"
            },
            "Holstein_Friesian": {
                "origin": "Netherlands",
                "type": "Dairy",
                "characteristics": "Highest milk production, black and white",
                "color": "Black and white patches"
            },
            "Jersey": {
                "origin": "Jersey Island",
                "type": "Dairy",
                "characteristics": "High butterfat content, small size",
                "color": "Light brown to dark brown"
            },
            "Murrah": {
                "origin": "Haryana, India",
                "type": "Dairy (Buffalo)",
                "characteristics": "High milk yield, black color, curved horns",
                "color": "Black"
            },
            "Ongole": {
                "origin": "Andhra Pradesh, India",
                "type": "Dual purpose",
                "characteristics": "Large, white, strong draft animal",
                "color": "White"
            },
            "Kankrej": {
                "origin": "Gujarat, India",
                "type": "Draft",
                "characteristics": "Large, strong, grey color",
                "color": "Grey"
            },
            "Tharparkar": {
                "origin": "Rajasthan, India",
                "type": "Dairy",
                "characteristics": "Hardy, drought resistant, white to light grey",
                "color": "White to light grey"
            },
            "Red_Sindhi": {
                "origin": "Sindh, Pakistan",
                "type": "Dairy",
                "characteristics": "Heat tolerant, good milk production",
                "color": "Deep red"
            }
        }
    
    def _normalize_text(self, text: str) -> str:
        """Normalize input text for better matching."""
        text = text.lower().strip()
        # Replace underscores with spaces for breed names
        text = text.replace("_", " ")
        return text
    
    def _extract_breed_name(self, query: str) -> str | None:
        """Extract breed name from query."""
        query_lower = self._normalize_text(query)
        
        # Direct match
        for breed in self.class_names:
            breed_normalized = self._normalize_text(breed)
            if breed_normalized in query_lower or query_lower in breed_normalized:
                return breed
        
        # Partial match
        for breed in self.class_names:
            breed_words = self._normalize_text(breed).split()
            if any(word in query_lower for word in breed_words if len(word) > 3):
                return breed
        
        return None
    
    def _is_greeting(self, query: str) -> bool:
        """Check if query is a greeting."""
        query_lower = self._normalize_text(query)
        return any(greeting in query_lower for greeting in self.greetings)
    
    def _is_help_request(self, query: str) -> bool:
        """Check if query is asking for help."""
        query_lower = self._normalize_text(query)
        return any(keyword in query_lower for keyword in self.help_keywords)
    
    def _get_breed_info(self, breed: str) -> str:
        """Get information about a specific breed."""
        breed_normalized = breed.replace("_", " ")
        info = self.breed_info.get(breed, {})
        
        if info:
            response = f"**{breed_normalized}**\n\n"
            response += f"📍 **Origin:** {info.get('origin', 'Unknown')}\n"
            response += f"🐄 **Type:** {info.get('type', 'Unknown')}\n"
            response += f"✨ **Characteristics:** {info.get('characteristics', 'Unknown')}\n"
            response += f"🎨 **Color:** {info.get('color', 'Unknown')}\n"
        else:
            response = f"**{breed_normalized}** is one of the cattle breeds in our database. "
            response += "I have limited information about this breed. Would you like to know about other breeds like Gir, Sahiwal, or Holstein Friesian?"
        
        return response
    
    def _get_all_breeds(self) -> str:
        """Get list of all available breeds."""
        breeds_list = [breed.replace("_", " ") for breed in self.class_names]
        response = f"I can help you with information about **{len(breeds_list)} cattle breeds**:\n\n"
        response += ", ".join(breeds_list[:10]) + "..."
        response += f"\n\nAsk me about any specific breed for detailed information!"
        return response
    
    def respond(self, query: str) -> str:
        """Generate response to user query."""
        if not query or not query.strip():
            return "Hello! I'm your cattle breed assistant. How can I help you today?"
        
        query_lower = self._normalize_text(query)
        
        # Handle greetings
        if self._is_greeting(query):
            return "Hello! 👋 I'm an AI assistant that can help you learn about cattle breeds. " \
                   "You can ask me about specific breeds, their characteristics, origins, or just say 'list breeds' to see all available breeds!"
        
        # Handle help requests
        if self._is_help_request(query):
            return "**I can help you with:**\n\n" \
                   "🐄 Information about specific cattle breeds\n" \
                   "📍 Origin and characteristics of breeds\n" \
                   "📋 List all available breeds\n" \
                   "💡 General questions about cattle breeds\n\n" \
                   "Try asking: 'Tell me about Gir cattle' or 'What breeds do you know?'"
        
        # Check for breed-specific queries
        breed = self._extract_breed_name(query)
        if breed:
            return self._get_breed_info(breed)
        
        # Check for list requests
        if any(word in query_lower for word in ["list", "all breeds", "available", "show breeds"]):
            return self._get_all_breeds()
        
        # Check for general questions
        if any(word in query_lower for word in ["dairy", "milk"]):
            dairy_breeds = ["Gir", "Sahiwal", "Holstein_Friesian", "Jersey", "Murrah", "Tharparkar"]
            return f"**Dairy breeds** in our database include: {', '.join([b.replace('_', ' ') for b in dairy_breeds])}. " \
                   "Ask me about any specific breed for more details!"
        
        if any(word in query_lower for word in ["draft", "work", "agricultural"]):
            draft_breeds = ["Alambadi", "Ongole", "Kankrej"]
            return f"**Draft breeds** (used for work) include: {', '.join([b.replace('_', ' ') for b in draft_breeds])}. " \
                   "These breeds are known for their strength and endurance."
        
        # Default response
        return "I'm not sure I understand that question. 🤔 " \
               "I can help you with information about cattle breeds. " \
               "Try asking:\n" \
               "- 'Tell me about [breed name]'\n" \
               "- 'List all breeds'\n" \
               "- 'What dairy breeds do you know?'\n" \
               "- Or just say 'help' for more options!"


def chat_interface():
    """Simple command-line chat interface."""
    chatbot = CattleBreedChatbot()
    print("🐄 Cattle Breed AI Chatbot")
    print("Type 'quit' or 'exit' to end the conversation\n")
    
    while True:
        user_input = input("You: ").strip()
        if user_input.lower() in ['quit', 'exit', 'bye']:
            print("Chatbot: Goodbye! Have a great day! 👋")
            break
        
        response = chatbot.respond(user_input)
        print(f"Chatbot: {response}\n")


if __name__ == "__main__":
    chat_interface()

