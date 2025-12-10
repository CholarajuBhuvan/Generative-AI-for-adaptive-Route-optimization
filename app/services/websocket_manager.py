"""
WebSocket Manager for Real-time Route Updates
"""

import asyncio
import json
from typing import Dict, List, Set
from fastapi import WebSocket, WebSocketDisconnect
import logging


class WebSocketManager:
    """Manages WebSocket connections for real-time updates"""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.connection_subscriptions: Dict[str, Set[str]] = {}  # connection_id -> set of topics
        self.topic_subscribers: Dict[str, Set[str]] = {}  # topic -> set of connection_ids
    
    async def connect(self, websocket: WebSocket, connection_id: str = None):
        """Accept a new WebSocket connection"""
        await websocket.accept()
        
        if connection_id is None:
            connection_id = f"conn_{len(self.active_connections)}_{id(websocket)}"
        
        self.active_connections[connection_id] = websocket
        self.connection_subscriptions[connection_id] = set()
        
        logging.info(f"WebSocket connected: {connection_id}")
        
        # Send welcome message
        await self.send_personal_message({
            "type": "connection_established",
            "connection_id": connection_id,
            "message": "Connected to AI Route Optimization System"
        }, connection_id)
    
    def disconnect(self, websocket: WebSocket):
        """Remove a WebSocket connection"""
        # Find connection ID
        connection_id = None
        for conn_id, conn in self.active_connections.items():
            if conn == websocket:
                connection_id = conn_id
                break
        
        if connection_id:
            # Remove from all topic subscriptions
            if connection_id in self.connection_subscriptions:
                for topic in self.connection_subscriptions[connection_id]:
                    if topic in self.topic_subscribers:
                        self.topic_subscribers[topic].discard(connection_id)
                        if not self.topic_subscribers[topic]:
                            del self.topic_subscribers[topic]
                
                del self.connection_subscriptions[connection_id]
            
            # Remove connection
            del self.active_connections[connection_id]
            logging.info(f"WebSocket disconnected: {connection_id}")
    
    async def send_personal_message(self, message: Dict, connection_id: str):
        """Send message to a specific connection"""
        if connection_id in self.active_connections:
            websocket = self.active_connections[connection_id]
            try:
                await websocket.send_text(json.dumps(message))
            except Exception as e:
                logging.error(f"Error sending message to {connection_id}: {e}")
                self.disconnect(websocket)
    
    async def broadcast_to_topic(self, message: Dict, topic: str):
        """Broadcast message to all subscribers of a topic"""
        if topic in self.topic_subscribers:
            for connection_id in list(self.topic_subscribers[topic]):
                await self.send_personal_message(message, connection_id)
    
    async def broadcast_to_all(self, message: Dict):
        """Broadcast message to all active connections"""
        for connection_id in list(self.active_connections.keys()):
            await self.send_personal_message(message, connection_id)
    
    async def subscribe_to_topic(self, connection_id: str, topic: str):
        """Subscribe a connection to a topic"""
        if connection_id in self.active_connections:
            # Add to connection's subscriptions
            if connection_id not in self.connection_subscriptions:
                self.connection_subscriptions[connection_id] = set()
            self.connection_subscriptions[connection_id].add(topic)
            
            # Add to topic's subscribers
            if topic not in self.topic_subscribers:
                self.topic_subscribers[topic] = set()
            self.topic_subscribers[topic].add(connection_id)
            
            # Send confirmation
            await self.send_personal_message({
                "type": "subscription_confirmed",
                "topic": topic,
                "message": f"Subscribed to {topic}"
            }, connection_id)
    
    async def unsubscribe_from_topic(self, connection_id: str, topic: str):
        """Unsubscribe a connection from a topic"""
        if connection_id in self.active_connections:
            # Remove from connection's subscriptions
            if connection_id in self.connection_subscriptions:
                self.connection_subscriptions[connection_id].discard(topic)
            
            # Remove from topic's subscribers
            if topic in self.topic_subscribers:
                self.topic_subscribers[topic].discard(connection_id)
                if not self.topic_subscribers[topic]:
                    del self.topic_subscribers[topic]
            
            # Send confirmation
            await self.send_personal_message({
                "type": "unsubscription_confirmed",
                "topic": topic,
                "message": f"Unsubscribed from {topic}"
            }, connection_id)
    
    async def handle_message(self, websocket: WebSocket, message: str):
        """Handle incoming WebSocket messages"""
        try:
            data = json.loads(message)
            message_type = data.get("type")
            
            # Find connection ID
            connection_id = None
            for conn_id, conn in self.active_connections.items():
                if conn == websocket:
                    connection_id = conn_id
                    break
            
            if not connection_id:
                return
            
            if message_type == "subscribe":
                topic = data.get("topic")
                if topic:
                    await self.subscribe_to_topic(connection_id, topic)
            
            elif message_type == "unsubscribe":
                topic = data.get("topic")
                if topic:
                    await self.unsubscribe_from_topic(connection_id, topic)
            
            elif message_type == "ping":
                await self.send_personal_message({
                    "type": "pong",
                    "timestamp": data.get("timestamp")
                }, connection_id)
            
            elif message_type == "get_status":
                await self.send_personal_message({
                    "type": "status",
                    "active_connections": len(self.active_connections),
                    "topics": list(self.topic_subscribers.keys()),
                    "connection_id": connection_id
                }, connection_id)
            
            else:
                await self.send_personal_message({
                    "type": "error",
                    "message": f"Unknown message type: {message_type}"
                }, connection_id)
                
        except json.JSONDecodeError:
            await self.send_personal_message({
                "type": "error",
                "message": "Invalid JSON format"
            }, websocket)
        except Exception as e:
            logging.error(f"Error handling WebSocket message: {e}")
            await self.send_personal_message({
                "type": "error",
                "message": "Internal server error"
            }, websocket)
    
    def get_connection_stats(self) -> Dict:
        """Get connection statistics"""
        return {
            "active_connections": len(self.active_connections),
            "total_topics": len(self.topic_subscribers),
            "topics": {
                topic: len(subscribers) 
                for topic, subscribers in self.topic_subscribers.items()
            }
        }

# Global instance for application-wide use
websocket_manager = WebSocketManager()
