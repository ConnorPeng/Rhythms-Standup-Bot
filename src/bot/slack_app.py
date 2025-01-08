from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest
from typing import Dict, Any
import os
import logging
from langchain_core.messages import HumanMessage
from conversation.graph import create_standup_graph, StandupState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandupBot:
    def __init__(self):
        """Initialize the Standup Bot with Slack credentials."""
        self.app_token = os.getenv("SLACK_APP_TOKEN")
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        
        if not self.app_token or not self.bot_token:
            raise ValueError("SLACK_APP_TOKEN and SLACK_BOT_TOKEN must be set")
            
        # Initialize Slack clients
        self.client = WebClient(token=self.bot_token)
        self.socket_client = SocketModeClient(
            app_token=self.app_token,
            web_client=self.client
        )
        
        # Initialize conversation graph
        self.conversation_graph = create_standup_graph()
        
        # Set up message handler
        self._setup_handler()
        logger.info("StandupBot initialized successfully")
        
    def _setup_handler(self) -> None:
        """Setup the socket mode event handler."""
        def socket_handler(client: SocketModeClient, req: SocketModeRequest) -> None:
            if req.type == "events_api":
                # Acknowledge request
                response = SocketModeResponse(envelope_id=req.envelope_id)
                client.send_socket_mode_response(response)
                
                # Process event
                event = req.payload.get("event", {})
                
                # Handle only message events that aren't from bots
                if event.get("type") == "message" and "bot_id" not in event:
                    self._handle_message(event)

        self.socket_client.socket_mode_request_listeners.append(socket_handler)
        logger.info("Socket handler setup complete")

    def _handle_message(self, event: Dict[str, Any]) -> None:
        """Handle incoming messages."""
        channel_id = event["channel"]
        user_id = event["user"]
        text = event.get("text", "").strip()

        if "standup" in text.lower():
            self._generate_standup(channel_id, user_id, text)

    def _generate_standup(self, channel_id: str, user_id: str, text: str) -> None:
        """Generate standup update using LangGraph."""
        try:
            # Get user info
            user_info = self.client.users_info(user=user_id)["user"]
            
            # Create initial state
            initial_state = StandupState(
                messages=[HumanMessage(content=text)],
                user_info={
                    "id": user_id,
                    "name": user_info["real_name"],
                    "github_username": user_info["profile"].get("title", ""),
                },
                current_draft={},
                activities={},
                next_step="initialize"
            )
            
            # Run the graph
            result = self.conversation_graph.invoke(initial_state)
            
            # Send the response
            if result["messages"]:
                last_message = result["messages"][-1]
                self.client.chat_postMessage(
                    channel=channel_id,
                    text=last_message.content
                )
                
        except Exception as e:
            logger.error(f"Error generating standup: {str(e)}")
            self.client.chat_postMessage(
                channel=channel_id,
                text="Sorry, I encountered an error. Please try again."
            )

    def start(self) -> None:
        """Start the Slack bot."""
        logger.info("Starting Standup bot...")
        self.socket_client.connect()
        
        # Keep the bot running
        from time import sleep
        while True:
            sleep(1)

if __name__ == "__main__":
    bot = SimpleStandupBot()
    bot.start()