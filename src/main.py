import asyncio
import os
from dotenv import load_dotenv
from bot.slack_app import StandupBot

async def main():
    """Main application entry point."""
    # Load environment variables
    load_dotenv()
    
    # Create the bot
    bot = StandupBot()
    
    try:
        await bot.start()
    except KeyboardInterrupt:
        print("\nShutting down gracefully...")
        await bot.stop()
    except Exception as e:
        print(f"Error: {e}")
        raise

if __name__ == "__main__":
    asyncio.run(main())
