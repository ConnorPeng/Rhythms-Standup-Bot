from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest
from typing import Dict, Any
import os
from datetime import datetime
import json
import asyncio
from langchain_core.messages import HumanMessage, AIMessage
from conversation.graph import create_standup_graph, StandupState
import ssl

class StandupBot:
    def __init__(self):
        self.app_token = os.getenv("SLACK_APP_TOKEN")
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        print('connor debugging SLACK_BOT_TOKEN', os.getenv('SLACK_BOT_TOKEN'))
        print('connor debugging SLACK_APP_TOKEN', os.getenv('SLACK_APP_TOKEN'))
        if not self.app_token or not self.bot_token:
            raise ValueError("SLACK_APP_TOKEN and SLACK_BOT_TOKEN must be set in .env file")
            
        # Create SSL context
        ssl_context = ssl.create_default_context()
        ssl_context.check_hostname = False
        ssl_context.verify_mode = ssl.CERT_NONE
        
        # Initialize client with SSL context and base URL
        self.client = WebClient(
            token=self.bot_token,
            ssl=ssl_context,
            base_url="https://slack.com/api/"
        )

        try:
            response = self.client.auth_test()
            print("Bot is authenticated:", response)
        except Exception as e:
            print("Authentication failed:", e)
            print("Please check your SLACK_BOT_TOKEN and SLACK_APP_TOKEN in .env file")
            print("SLACK_BOT_TOKEN should start with 'xoxb-'")
            print("SLACK_APP_TOKEN should start with 'xapp-'")
            raise

        self.socket_client = SocketModeClient(
            app_token=self.app_token,
            web_client=self.client
        )
        
        # Define the handler as an async function
        async def socket_mode_request_handler(client, req):
            await self._process_socket_request(client, req)

        # Define a synchronous wrapper for the handler
        def sync_socket_handler(client, req):
            loop = asyncio.get_event_loop()
            if loop.is_running():
                loop.create_task(self._process_socket_request(client, req))
            else:
                loop.run_until_complete(self._process_socket_request(client, req))

        # Add the synchronous handler as a listener
        self.socket_client.socket_mode_request_listeners.append(sync_socket_handler)
        
        self.conversation_graph = create_standup_graph()
        self.active_conversations: Dict[str, Any] = {}

    async def _process_socket_request(self, client, req):
        """Wrapper to process socket requests asynchronously."""
        await self.handle_socket_request(client, req)

    async def start(self):
        """Start the Slack bot."""
        print("Standup bot is running...")
        # Start the client in a non-blocking way
        self.socket_client.connect()
        try:
            # Keep the bot running
            while True:
                await asyncio.sleep(1)
        except Exception as e:
            print(f"Error in bot: {e}")
            self.socket_client.close()
            raise

    async def handle_socket_request(
        self,
        client: SocketModeClient,
        req: SocketModeRequest
    ):
        """Handle incoming socket mode requests."""
        # Log the incoming request details
        print("\n=== Incoming Slack Request ===")
        print(f"Request Type: {req.type}")
        print(f"Envelope ID: {req.envelope_id}")
        print(f"Payload: {json.dumps(req.payload, indent=2)}")
        print("============================\n")

        if req.type == "events_api":
            # Acknowledge the request
            response = SocketModeResponse(envelope_id=req.envelope_id)
            client.send_socket_mode_response(response)
            
            # Process the event
            event = req.payload["event"]
            if event["type"] == "message" and "bot_id" not in event:
                await self._handle_message(event)
        
        elif req.type == "interactive":
            # Handle interactive components
            await self._handle_interactive(req)

    async def _handle_message(self, event: Dict[str, Any]):
        """Handle incoming messages."""
        # Log message event details
        print("\n=== Message Event Details ===")
        print(f"Channel ID: {event['channel']}")
        print(f"User ID: {event['user']}")
        print(f"Message Text: {event.get('text', '').strip()}")
        print(f"Timestamp: {event.get('ts', '')}")
        print("===========================\n")

        channel_id = event["channel"]
        user_id = event["user"]
        text = event.get("text", "").strip()

        if "standup" in text.lower():
            await self._initiate_standup(channel_id, user_id)
        elif user_id in self.active_conversations:
            await self._continue_conversation(channel_id, user_id, text)

    async def _handle_interactive(self, req: SocketModeRequest):
        """Handle interactive components."""
        # Log interactive event details
        print("\n=== Interactive Event Details ===")
        print(f"Type: {req.payload.get('type', '')}")
        print(f"Action ID: {req.payload.get('actions', [{}])[0].get('action_id', '')}")
        print(f"User: {req.payload.get('user', {}).get('id', '')}")
        print(f"Channel: {req.payload.get('channel', {}).get('id', '')}")
        print("==============================\n")

        # Rest of the function remains the same

    async def _initiate_standup(self, channel_id: str, user_id: str):
        """Start a new standup conversation."""
        try:
            # Get user info
            user_info = self.client.users_info(user=user_id)["user"]
            
            # Initialize conversation state
            initial_state = StandupState(
                messages=[
                    HumanMessage(content="Let's start my standup update.")
                ],
                user_info={
                    "id": user_id,
                    "name": user_info["real_name"],
                    "github_username": user_info["profile"].get("title", ""),
                },
                current_draft={},
                activities={},
                next_step="initialize"
            )
            
            # Store conversation state
            self.active_conversations[user_id] = {
                "state": initial_state,
                "channel": channel_id
            }
            
            # Run the conversation graph
            await self._run_graph(user_id)
            
        except Exception as e:
            print(f"Error initiating standup: {e}")
            self._send_error_message(channel_id)

    async def _continue_conversation(
        self,
        channel_id: str,
        user_id: str,
        text: str
    ):
        """Continue an existing conversation."""
        try:
            # Update conversation state with user's message
            conv_state = self.active_conversations[user_id]["state"]
            conv_state.messages.append(HumanMessage(content=text))
            
            # Run the conversation graph
            await self._run_graph(user_id)
            
        except Exception as e:
            print(f"Error continuing conversation: {e}")
            self._send_error_message(channel_id)

    async def _run_graph(self, user_id: str):
        """Run the conversation graph with current state."""
        try:
            conv_data = self.active_conversations[user_id]
            channel_id = conv_data["channel"]
            
            # Run the graph
            result = await self.conversation_graph.arun(
                conv_data["state"]
            )
            
            # Update conversation state
            self.active_conversations[user_id]["state"] = result
            
            # Send any new messages to the user
            messages = result.messages
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    await self.client.chat_postMessage(
                        channel=channel_id,
                        text=last_message.content
                    )
            
            # Check if conversation is complete
            if result.next_step == "end":
                self._finalize_conversation(user_id)
                
        except Exception as e:
            print(f"Error running conversation graph: {e}")
            self._send_error_message(channel_id)

    def _finalize_conversation(self, user_id: str):
        """Clean up after conversation is complete."""
        if user_id in self.active_conversations:
            # Post final update to team channel
            final_state = self.active_conversations[user_id]["state"]
            channel_id = self.active_conversations[user_id]["channel"]
            
            # Get the last message which should be the formatted update
            last_message = final_state.messages[-1].content
            
            # Post to channel
            asyncio.create_task(
                self.client.chat_postMessage(
                    channel=channel_id,
                    text=last_message
                )
            )
            
            # Clean up
            del self.active_conversations[user_id]

    def _send_error_message(self, channel_id: str):
        """Send an error message to the channel."""
        asyncio.create_task(
            self.client.chat_postMessage(
                channel=channel_id,
                text="Sorry, I encountered an error. Please try again."
            )
        )

if __name__ == "__main__":
    bot = StandupBot()
    asyncio.run(bot.start())

