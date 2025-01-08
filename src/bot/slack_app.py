from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest
from typing import Dict, Any, Optional
import os
import json
import asyncio
import logging
from langchain_core.messages import HumanMessage, AIMessage
from conversation.graph import create_standup_graph, StandupState

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class StandupBot:
    def __init__(self):
        """Initialize the Standup Bot with Slack credentials and setup socket mode client."""
        self.app_token = os.getenv("SLACK_APP_TOKEN")
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        
        if not self.app_token or not self.bot_token:
            logger.error("SLACK_APP_TOKEN or SLACK_BOT_TOKEN is missing!")
            raise ValueError("SLACK_APP_TOKEN and SLACK_BOT_TOKEN must be set in .env file")
            
        # Initialize Slack client
        self.client = WebClient(token=self.bot_token)
        self._authenticate_bot()
        
        # Initialize socket mode client
        self.socket_client = SocketModeClient(
            app_token=self.app_token,
            web_client=self.client
        )
        self._setup_socket_handler()
        
        # Initialize conversation state
        self.conversation_graph = create_standup_graph()
        self.active_conversations: Dict[str, Dict[str, Any]] = {}
        logger.debug("StandupBot initialized successfully.")

    def _authenticate_bot(self) -> None:
        """Authenticate the bot with Slack."""
        try:
            response = self.client.auth_test()
            logger.info(f"Bot authenticated successfully: {response['bot_id']}")
        except Exception as e:
            logger.error(f"Authentication failed: {str(e)}")
            raise

    def _setup_socket_handler(self) -> None:
        """Setup the socket mode event handler."""
        logger.debug("Setting up socket handler.")
        def sync_socket_handler(client: SocketModeClient, req: SocketModeRequest) -> None:
            logger.debug(f"Socket request received: {req.type}")
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(self._process_socket_request(client, req))
            except Exception as e:
                logger.error(f"Error processing socket request: {e}")
            finally:
                loop.close()

        self.socket_client.socket_mode_request_listeners.append(sync_socket_handler)
        logger.debug("Socket handler setup complete.")

    async def start(self) -> None:
        """Start the Slack bot."""
        logger.info("Starting Standup bot...")
        self.socket_client.connect()
        try:
            while True:
                logger.debug("Bot is running...")
                await asyncio.sleep(1)
        except Exception as e:
            logger.error(f"Error in bot: {str(e)}")
            self.socket_client.close()
            raise

    async def _process_socket_request(
        self,
        client: SocketModeClient,
        req: SocketModeRequest
    ) -> None:
        """Process incoming socket mode requests."""
        logger.debug(f"Processing socket request: {req.payload}")
        await self.handle_socket_request(client, req)

    async def handle_socket_request(
        self,
        client: SocketModeClient,
        req: SocketModeRequest
    ) -> None:
        """Handle incoming socket mode requests."""
        logger.debug(f"Handling socket request of type: {req.type}")
        
        if req.type == "events_api":
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)
            logger.debug("Acknowledged events_api request.")

            event = req.payload.get("event", {})
            logger.debug(f"Event payload: {event}")
            if event.get("type") == "message" and "bot_id" not in event:
                await self._handle_message(event)

    async def _handle_message(self, event: Dict[str, Any]) -> None:
        """Handle incoming messages."""
        channel_id = event["channel"]
        user_id = event["user"]
        text = event.get("text", "").strip()

        logger.info(f"Message received from {user_id} in {channel_id}: {text}")

        if "standup" in text.lower():
            logger.debug("Detected 'standup' in message. Initiating standup.")
            await self._initiate_standup(channel_id, user_id)
        elif user_id in self.active_conversations:
            logger.debug(f"Continuing conversation with user {user_id}.")
            await self._continue_conversation(channel_id, user_id, text)

    async def _initiate_standup(self, channel_id: str, user_id: str) -> None:
        """Start a new standup conversation."""
        logger.debug(f"Initiating standup for user {user_id} in channel {channel_id}.")
        try:
            user_info = self.client.users_info(user=user_id)["user"]
            logger.debug(f"User info fetched: {user_info}")
            
            initial_state = StandupState(
                messages=[HumanMessage(content="Let's start my standup update.")],
                user_info={
                    "id": user_id,
                    "name": user_info["real_name"],
                    "github_username": user_info["profile"].get("title", ""),
                },
                current_draft={},
                activities={},
                next_step="initialize"
            )
            
            self.active_conversations[user_id] = {
                "state": initial_state,
                "channel": channel_id
            }
            
            await self._run_graph(user_id)
            
        except Exception as e:
            logger.error(f"Error initiating standup: {str(e)}")
            await self._send_error_message(channel_id)

    async def _run_graph(self, user_id: str) -> None:
        """Run the conversation graph with current state."""
        logger.debug(f"Running conversation graph for user {user_id}.")
        try:
            conv_data = self.active_conversations[user_id]
            channel_id = conv_data["channel"]
            
            result = await self.conversation_graph.arun(conv_data["state"])
            logger.debug(f"Graph result: {result}")
            self.active_conversations[user_id]["state"] = result
            
            if result.messages:
                last_message = result.messages[-1]
                if isinstance(last_message, AIMessage):
                    logger.info(f"Sending message to channel {channel_id}: {last_message.content}")
                    await self.client.chat_postMessage(
                        channel=channel_id,
                        text=last_message.content
                    )
            
            if result.next_step == "end":
                logger.debug(f"Conversation with user {user_id} ended.")
                await self._finalize_conversation(user_id)
                
        except Exception as e:
            logger.error(f"Error running conversation graph: {str(e)}")
            await self._send_error_message(conv_data["channel"])


    async def _finalize_conversation(self, user_id: str) -> None:
        """Clean up after conversation is complete."""
        if user_id in self.active_conversations:
            final_state = self.active_conversations[user_id]["state"]
            channel_id = self.active_conversations[user_id]["channel"]
            
            last_message = final_state.messages[-1].content
            await self.client.chat_postMessage(
                channel=channel_id,
                text=last_message
            )
            
            del self.active_conversations[user_id]

    async def _send_error_message(self, channel_id: str) -> None:
        """Send an error message to the channel."""
        await self.client.chat_postMessage(
            channel=channel_id,
            text="Sorry, I encountered an error. Please try again."
        )

if __name__ == "__main__":
    bot = StandupBot()
    asyncio.run(bot.start())
