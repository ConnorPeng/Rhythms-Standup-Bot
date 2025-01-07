from slack_sdk.web import WebClient
from slack_sdk.socket_mode import SocketModeClient
from slack_sdk.socket_mode.response import SocketModeResponse
from slack_sdk.socket_mode.request import SocketModeRequest
from typing import Dict, Any
import os
from datetime import datetime
import json
import asyncio
from ..conversation.graph import create_standup_graph, StandupState
from langchain_core.messages import HumanMessage, AIMessage

class StandupBot:
    def __init__(self):
        self.app_token = os.getenv("SLACK_APP_TOKEN")
        self.bot_token = os.getenv("SLACK_BOT_TOKEN")
        self.client = WebClient(token=self.bot_token)
        self.socket_client = SocketModeClient(
            app_token=self.app_token,
            web_client=self.client
        )
        self.conversation_graph = create_standup_graph()
        self.active_conversations: Dict[str, Any] = {}

    async def start(self):
        """Start the Slack bot."""
        self.socket_client.socket_mode_request_listeners.append(
            self.handle_socket_request
        )
        self.socket_client.connect()
        print("Standup bot is running...")
        
        # Keep the bot running
        while True:
            await asyncio.sleep(1)

    async def handle_socket_request(
        self,
        client: SocketModeClient,
        req: SocketModeRequest
    ):
        """Handle incoming socket mode requests."""
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
        channel_id = event["channel"]
        user_id = event["user"]
        text = event.get("text", "").strip()

        if "standup" in text.lower():
            await self._initiate_standup(channel_id, user_id)
        elif user_id in self.active_conversations:
            await self._continue_conversation(channel_id, user_id, text)

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
            conv_state["messages"].append(HumanMessage(content=text))
            
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
            messages = result["messages"]
            if messages:
                last_message = messages[-1]
                if isinstance(last_message, AIMessage):
                    self.client.chat_postMessage(
                        channel=channel_id,
                        text=last_message.content
                    )
            
            # Check if conversation is complete
            if result["next_step"] == "end":
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
            last_message = final_state["messages"][-1].content
            
            # Post to channel
            self.client.chat_postMessage(
                channel=channel_id,
                text=last_message
            )
            
            # Clean up
            del self.active_conversations[user_id]

    def _send_error_message(self, channel_id: str):
        """Send an error message to the channel."""
        self.client.chat_postMessage(
            channel=channel_id,
            text="Sorry, I encountered an error. Please try again."
        )
