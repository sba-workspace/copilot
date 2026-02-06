#!/usr/bin/env python3
"""
MCP Server Launcher
Starts both GitHub and Notion MCP servers
"""

import asyncio
import logging
import subprocess
import sys
from pathlib import Path

from app.mcp.config import get_server_config, validate_environment

logger = logging.getLogger(__name__)

class MCPServerManager:
    """Manages MCP server processes"""
    
    def __init__(self):
        self.processes = {}
        self.config = get_server_config()
    
    async def start_github_server(self):
        """Start GitHub MCP server"""
        if not self.config["github"]["enabled"]:
            logger.warning("GitHub MCP server disabled (GITHUB_TOKEN not set)")
            return False
        
        try:
            cmd = [
                sys.executable, "-m", "app.mcp.github_mcp_server"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.processes["github"] = process
            logger.info(f"GitHub MCP server started on port {self.config['github']['port']}")
            return True
        except Exception as e:
            logger.error(f"Failed to start GitHub MCP server: {str(e)}")
            return False
    
    async def start_notion_server(self):
        """Start Notion MCP server"""
        if not self.config["notion"]["enabled"]:
            logger.warning("Notion MCP server disabled (NOTION_API_KEY not set)")
            return False
        
        try:
            cmd = [
                sys.executable, "-m", "app.mcp.notion_mcp_server"
            ]
            
            process = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            self.processes["notion"] = process
            logger.info(f"Notion MCP server started on port {self.config['notion']['port']}")
            return True
        except Exception as e:
            logger.error(f"Failed to start Notion MCP server: {str(e)}")
            return False
    
    async def start_all_servers(self):
        """Start all enabled MCP servers"""
        logger.info("Starting MCP servers...")
        
        if not validate_environment():
            logger.error("Environment validation failed")
            return False
        
        tasks = []
        
        if self.config["github"]["enabled"]:
            tasks.append(self.start_github_server())
        
        if self.config["notion"]["enabled"]:
            tasks.append(self.start_notion_server())
        
        if not tasks:
            logger.error("No MCP servers enabled")
            return False
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        success_count = sum(1 for result in results if result is True)
        logger.info(f"Started {success_count}/{len(tasks)} MCP servers")
        
        return success_count > 0
    
    async def stop_all_servers(self):
        """Stop all running MCP servers"""
        logger.info("Stopping MCP servers...")
        
        for service, process in self.processes.items():
            try:
                process.terminate()
                await process.wait()
                logger.info(f"{service} MCP server stopped")
            except Exception as e:
                logger.error(f"Error stopping {service} MCP server: {str(e)}")
        
        self.processes.clear()
    
    async def monitor_servers(self):
        """Monitor server health and restart if needed"""
        while self.processes:
            for service, process in list(self.processes.items()):
                if process.returncode is not None:
                    logger.warning(f"{service} MCP server died, restarting...")
                    if service == "github":
                        await self.start_github_server()
                    elif service == "notion":
                        await self.start_notion_server()
            
            await asyncio.sleep(5)

async def main():
    """Main entry point"""
    manager = MCPServerManager()
    
    try:
        success = await manager.start_all_servers()
        if not success:
            logger.error("Failed to start MCP servers")
            return 1
        
        logger.info("MCP servers running. Press Ctrl+C to stop.")
        await manager.monitor_servers()
        
    except KeyboardInterrupt:
        logger.info("Received interrupt signal")
    except Exception as e:
        logger.error(f"Unexpected error: {str(e)}")
        return 1
    finally:
        await manager.stop_all_servers()
    
    return 0

if __name__ == "__main__":
    exit_code = asyncio.run(main())
    sys.exit(exit_code)
