#!/usr/bin/env python3
"""
GitHub Multi-Agent Collaborative Protocol (MCP)
==============================================
A framework for enabling agent-to-agent communication and collaboration through GitHub.
Agents can work together on repositories, coordinate tasks, and maintain shared state.
"""

import os
import json
import time
import base64
import requests
from datetime import datetime
from typing import Dict, List, Any, Optional, Union, Tuple

class GitHubMCP:
    """Main class for the GitHub Multi-Agent Collaborative Protocol."""
    
    def __init__(self, token: str, agent_id: str, repo_owner: str, repo_name: str):
        """
        Initialize the GitHub MCP with authentication and repository details.
        
        Args:
            token: GitHub Personal Access Token with appropriate permissions
            agent_id: Unique identifier for this agent
            repo_owner: Owner of the GitHub repository
            repo_name: Name of the GitHub repository
        """
        self.token = token
        self.agent_id = agent_id
        self.repo_owner = repo_owner
        self.repo_name = repo_name
        self.base_url = "https://api.github.com"
        self.headers = {
            "Authorization": f"token {token}",
            "Accept": "application/vnd.github.v3+json"
        }
        self.comm_branch = "mcp-communication"
        self.state_file = "mcp_state.json"
        self.message_dir = "mcp_messages"
        self.task_dir = "mcp_tasks"
        self._ensure_mcp_infrastructure()
    
    def _ensure_mcp_infrastructure(self) -> None:
        """Ensure all necessary branches and files exist for MCP operation."""
        # Check if communication branch exists, create if not
        try:
            self._make_request("GET", f"/repos/{self.repo_owner}/{self.repo_name}/branches/{self.comm_branch}")
        except Exception:
            # Get default branch to base our branch on
            default_branch = self._get_default_branch()
            self._create_branch(self.comm_branch, default_branch)
        
        # Ensure state file exists
        try:
            self._get_file_content(self.state_file, self.comm_branch)
        except Exception:
            # Create initial state file
            initial_state = {
                "agents": {self.agent_id: {"last_active": self._get_timestamp()}},
                "tasks": {},
                "shared_data": {}
            }
            self._create_or_update_file(
                self.state_file,
                f"Initialize MCP state file for agent {self.agent_id}",
                json.dumps(initial_state, indent=2),
                self.comm_branch
            )
        
        # Ensure message and task directories exist
        for directory in [self.message_dir, self.task_dir]:
            try:
                self._make_request("GET", f"/repos/{self.repo_owner}/{self.repo_name}/contents/{directory}",
                                params={"ref": self.comm_branch})
            except Exception:
                # Create directory with a .gitkeep file
                self._create_or_update_file(
                    f"{directory}/.gitkeep",
                    f"Create {directory} directory for MCP",
                    "",
                    self.comm_branch
                )
    
    def _make_request(self, method: str, endpoint: str, **kwargs) -> Dict:
        """
        Make an authenticated request to the GitHub API.
        
        Args:
            method: HTTP method (GET, POST, PUT, DELETE, etc.)
            endpoint: API endpoint (starting with /)
            **kwargs: Additional arguments to pass to requests
            
        Returns:
            Dict: Response from the GitHub API
        """
        url = f"{self.base_url}{endpoint}"
        headers = kwargs.pop("headers", {})
        headers.update(self.headers)
        
        response = requests.request(method, url, headers=headers, **kwargs)
        
        if response.status_code >= 400:
            raise Exception(f"GitHub API error: {response.status_code}, {response.text}")
        
        return response.json() if response.text else {}
    
    def _get_default_branch(self) -> str:
        """Get the default branch of the repository."""
        repo_info = self._make_request("GET", f"/repos/{self.repo_owner}/{self.repo_name}")
        return repo_info["default_branch"]
    
    def _create_branch(self, branch_name: str, base_branch: str) -> None:
        """
        Create a new branch in the repository.
        
        Args:
            branch_name: Name of the new branch
            base_branch: Name of the branch to base the new branch on
        """
        # Get the SHA of the latest commit on the base branch
        base_ref = self._make_request(
            "GET", 
            f"/repos/{self.repo_owner}/{self.repo_name}/git/ref/heads/{base_branch}"
        )
        base_sha = base_ref["object"]["sha"]
        
        # Create the new branch reference
        self._make_request(
            "POST",
            f"/repos/{self.repo_owner}/{self.repo_name}/git/refs",
            json={"ref": f"refs/heads/{branch_name}", "sha": base_sha}
        )
    
    def _get_file_content(self, path: str, branch: str = None) -> Tuple[str, str]:
        """
        Get the content and SHA of a file from the repository.
        
        Args:
            path: Path to the file
            branch: Branch to get the file from (default: repository default branch)
            
        Returns:
            Tuple[str, str]: Content of the file and its SHA
        """
        params = {"ref": branch} if branch else {}
        file_info = self._make_request(
            "GET",
            f"/repos/{self.repo_owner}/{self.repo_name}/contents/{path}",
            params=params
        )
        
        if file_info["encoding"] == "base64":
            content = base64.b64decode(file_info["content"]).decode("utf-8")
        else:
            content = file_info["content"]
            
        return content, file_info["sha"]
    
    def _create_or_update_file(self, path: str, commit_message: str, content: str, branch: str = None) -> str:
        """
        Create or update a file in the repository.
        
        Args:
            path: Path to the file
            commit_message: Commit message
            content: Content to write to the file
            branch: Branch to create/update the file in (default: repository default branch)
            
        Returns:
            str: SHA of the new commit
        """
        data = {
            "message": commit_message,
            "content": base64.b64encode(content.encode("utf-8")).decode("utf-8"),
            "branch": branch
        }
        
        # Check if file exists to determine whether to create or update
        try:
            _, sha = self._get_file_content(path, branch)
            data["sha"] = sha  # Include SHA if updating existing file
        except Exception:
            pass  # File doesn't exist, so we're creating it
        
        response = self._make_request(
            "PUT",
            f"/repos/{self.repo_owner}/{self.repo_name}/contents/{path}",
            json=data
        )
        
        return response["commit"]["sha"]
    
    def _get_timestamp(self) -> str:
        """Get the current timestamp in ISO format."""
        return datetime.utcnow().isoformat() + "Z"
    
    def _update_state(self, update_func) -> Dict:
        """
        Update the shared state file using a provided update function.
        
        Args:
            update_func: Function that takes the current state and returns the updated state
            
        Returns:
            Dict: The updated state
        """
        # Get current state
        state_content, state_sha = self._get_file_content(self.state_file, self.comm_branch)
        current_state = json.loads(state_content)
        
        # Update agent's last active timestamp
        if self.agent_id not in current_state["agents"]:
            current_state["agents"][self.agent_id] = {}
        current_state["agents"][self.agent_id]["last_active"] = self._get_timestamp()
        
        # Apply custom updates
        updated_state = update_func(current_state)
        
        # Write back to repository
        self._create_or_update_file(
            self.state_file,
            f"Update MCP state via agent {self.agent_id}",
            json.dumps(updated_state, indent=2),
            self.comm_branch
        )
        
        return updated_state
    
    def register_agent(self, capabilities: List[str] = None, metadata: Dict = None) -> None:
        """
        Register this agent or update its information in the shared state.
        
        Args:
            capabilities: List of capabilities this agent has
            metadata: Additional metadata about this agent
        """
        def update_state(state):
            if self.agent_id not in state["agents"]:
                state["agents"][self.agent_id] = {}
            
            agent_info = state["agents"][self.agent_id]
            agent_info["last_active"] = self._get_timestamp()
            
            if capabilities:
                agent_info["capabilities"] = capabilities
                
            if metadata:
                if "metadata" not in agent_info:
                    agent_info["metadata"] = {}
                agent_info["metadata"].update(metadata)
                
            return state
        
        self._update_state(update_state)
    
    def get_active_agents(self, timeout_minutes: int = 60) -> Dict[str, Dict]:
        """
        Get all active agents (active within the specified timeout period).
        
        Args:
            timeout_minutes: Number of minutes after which an agent is considered inactive
            
        Returns:
            Dict[str, Dict]: Dictionary of active agent IDs to agent information
        """
        state_content, _ = self._get_file_content(self.state_file, self.comm_branch)
        state = json.loads(state_content)
        
        active_agents = {}
        current_time = datetime.fromisoformat(self._get_timestamp().rstrip("Z"))
        
        for agent_id, agent_info in state["agents"].items():
            last_active = datetime.fromisoformat(agent_info["last_active"].rstrip("Z"))
            time_diff = (current_time - last_active).total_seconds() / 60
            
            if time_diff <= timeout_minutes:
                active_agents[agent_id] = agent_info
                
        return active_agents
    
    def send_message(self, recipient_id: str, message_type: str, content: Dict) -> str:
        """
        Send a message to another agent.
        
        Args:
            recipient_id: ID of the recipient agent
            message_type: Type of message (task, response, query, etc.)
            content: Content of the message
            
        Returns:
            str: ID of the created message
        """
        message_id = f"{int(time.time())}_{self.agent_id}_{recipient_id}"
        message = {
            "id": message_id,
            "sender": self.agent_id,
            "recipient": recipient_id,
            "type": message_type,
            "timestamp": self._get_timestamp(),
            "content": content,
            "read": False
        }
        
        message_path = f"{self.message_dir}/{message_id}.json"
        self._create_or_update_file(
            message_path,
            f"Message from {self.agent_id} to {recipient_id}",
            json.dumps(message, indent=2),
            self.comm_branch
        )
        
        return message_id
    
    def get_messages(self, unread_only: bool = False) -> List[Dict]:
        """
        Get messages addressed to this agent.
        
        Args:
            unread_only: Whether to get only unread messages
            
        Returns:
            List[Dict]: List of messages
        """
        # Get list of all message files
        try:
            message_files = self._make_request(
                "GET",
                f"/repos/{self.repo_owner}/{self.repo_name}/contents/{self.message_dir}",
                params={"ref": self.comm_branch}
            )
        except Exception:
            return []
            
        messages = []
        for file_info in message_files:
            if file_info["type"] != "file" or not file_info["name"].endswith(".json"):
                continue
                
            try:
                content, sha = self._get_file_content(file_info["path"], self.comm_branch)
                message = json.loads(content)
                
                # Only process messages addressed to this agent
                if message["recipient"] == self.agent_id:
                    if not unread_only or not message["read"]:
                        message["_file_path"] = file_info["path"]
                        message["_file_sha"] = sha
                        messages.append(message)
            except Exception as e:
                print(f"Error processing message {file_info['name']}: {e}")
                
        return messages
    
    def mark_messages_read(self, message_ids: List[str]) -> None:
        """
        Mark messages as read.
        
        Args:
            message_ids: List of message IDs to mark as read
        """
        messages = self.get_messages(unread_only=True)
        
        for message in messages:
            if message["id"] in message_ids:
                message["read"] = True
                self._create_or_update_file(
                    message["_file_path"],
                    f"Mark message as read by {self.agent_id}",
                    json.dumps({k: v for k, v in message.items() if not k.startswith('_')}, indent=2),
                    self.comm_branch
                )
    
    def create_task(self, task_type: str, parameters: Dict, assigned_to: str = None,
                   priority: int = 1, deadline: str = None) -> str:
        """
        Create a new task in the shared task list.
        
        Args:
            task_type: Type of task
            parameters: Parameters for the task
            assigned_to: ID of the agent assigned to the task (or None for unassigned)
            priority: Priority of the task (1=low, 3=high)
            deadline: Optional deadline for the task (ISO format timestamp)
            
        Returns:
            str: ID of the created task
        """
        task_id = f"task_{int(time.time())}_{self.agent_id}"
        task = {
            "id": task_id,
            "type": task_type,
            "creator": self.agent_id,
            "assigned_to": assigned_to,
            "parameters": parameters,
            "status": "pending" if assigned_to else "unassigned",
            "priority": priority,
            "created_at": self._get_timestamp(),
            "deadline": deadline,
            "result": None
        }
        
        # Add to shared state
        def update_state(state):
            state["tasks"][task_id] = {
                "type": task_type,
                "status": task["status"],
                "assigned_to": assigned_to,
                "created_at": task["created_at"]
            }
            return state
            
        self._update_state(update_state)
        
        # Save detailed task info
        task_path = f"{self.task_dir}/{task_id}.json"
        self._create_or_update_file(
            task_path,
            f"Create task by {self.agent_id}",
            json.dumps(task, indent=2),
            self.comm_branch
        )
        
        return task_id
    
    def get_tasks(self, status: str = None, assigned_to: str = None) -> List[Dict]:
        """
        Get tasks from the task list.
        
        Args:
            status: Filter by status (pending, completed, failed, unassigned)
            assigned_to: Filter by assigned agent ID
            
        Returns:
            List[Dict]: List of matching tasks
        """
        try:
            task_files = self._make_request(
                "GET",
                f"/repos/{self.repo_owner}/{self.repo_name}/contents/{self.task_dir}",
                params={"ref": self.comm_branch}
            )
        except Exception:
            return []
            
        tasks = []
        for file_info in task_files:
            if file_info["type"] != "file" or not file_info["name"].endswith(".json"):
                continue
                
            try:
                content, _ = self._get_file_content(file_info["path"], self.comm_branch)
                task = json.loads(content)
                
                # Apply filters
                if status and task["status"] != status:
                    continue
                if assigned_to and task["assigned_to"] != assigned_to:
                    continue
                    
                tasks.append(task)
            except Exception as e:
                print(f"Error processing task {file_info['name']}: {e}")
                
        return tasks
    
    def update_task_status(self, task_id: str, status: str, result: Any = None) -> None:
        """
        Update the status of a task.
        
        Args:
            task_id: ID of the task to update
            status: New status (pending, completed, failed)
            result: Optional result data
        """
        task_path = f"{self.task_dir}/{task_id}.json"
        content, _ = self._get_file_content(task_path, self.comm_branch)
        task = json.loads(content)
        
        # Update task status
        task["status"] = status
        if result is not None:
            task["result"] = result
        if status in ["completed", "failed"]:
            task["completed_at"] = self._get_timestamp()
            
        self._create_or_update_file(
            task_path,
            f"Update task status to {status} by {self.agent_id}",
            json.dumps(task, indent=2),
            self.comm_branch
        )
        
        # Update in shared state as well
        def update_state(state):
            if task_id in state["tasks"]:
                state["tasks"][task_id]["status"] = status
            return state
            
        self._update_state(update_state)
    
    def assign_task(self, task_id: str, agent_id: str) -> None:
        """
        Assign a task to an agent.
        
        Args:
            task_id: ID of the task to assign
            agent_id: ID of the agent to assign the task to
        """
        task_path = f"{self.task_dir}/{task_id}.json"
        content, _ = self._get_file_content(task_path, self.comm_branch)
        task = json.loads(content)
        
        # Update task assignment
        task["assigned_to"] = agent_id
        if task["status"] == "unassigned":
            task["status"] = "pending"
            
        self._create_or_update_file(
            task_path,
            f"Assign task to {agent_id} by {self.agent_id}",
            json.dumps(task, indent=2),
            self.comm_branch
        )
        
        # Update in shared state as well
        def update_state(state):
            if task_id in state["tasks"]:
                state["tasks"][task_id]["assigned_to"] = agent_id
                state["tasks"][task_id]["status"] = task["status"]
            return state
            
        self._update_state(update_state)
    
    def set_shared_data(self, key: str, value: Any) -> None:
        """
        Set a value in the shared data store.
        
        Args:
            key: Key to store the value under
            value: Value to store (must be JSON-serializable)
        """
        def update_state(state):
            state["shared_data"][key] = value
            return state
            
        self._update_state(update_state)
    
    def get_shared_data(self, key: str = None) -> Any:
        """
        Get a value from the shared data store.
        
        Args:
            key: Key to retrieve (None to get all shared data)
            
        Returns:
            The value stored under the key, or all shared data if key is None
        """
        state_content, _ = self._get_file_content(self.state_file, self.comm_branch)
        state = json.loads(state_content)
        
        if key is None:
            return state["shared_data"]
        
        return state["shared_data"].get(key)
    
    def create_issue(self, title: str, body: str, labels: List[str] = None) -> Dict:
        """
        Create a GitHub issue in the repository.
        
        Args:
            title: Issue title
            body: Issue body/description
            labels: List of labels to apply to the issue
            
        Returns:
            Dict: Created issue information
        """
        data = {
            "title": title,
            "body": body
        }
        
        if labels:
            data["labels"] = labels
            
        return self._make_request(
            "POST",
            f"/repos/{self.repo_owner}/{self.repo_name}/issues",
            json=data
        )
    
    def create_pull_request(self, title: str, body: str, head_branch: str, base_branch: str) -> Dict:
        """
        Create a pull request.
        
        Args:
            title: Pull request title
            body: Pull request description
            head_branch: Branch containing the changes
            base_branch: Branch to merge the changes into
            
        Returns:
            Dict: Created pull request information
        """
        data = {
            "title": title,
            "body": body,
            "head": head_branch,
            "base": base_branch
        }
        
        return self._make_request(
            "POST",
            f"/repos/{self.repo_owner}/{self.repo_name}/pulls",
            json=data
        )


# Example usage
if __name__ == "__main__":
    token = os.environ.get("GITHUB_TOKEN", "your_token_here")
    
    # Create an instance of the GitHub MCP
    mcp = GitHubMCP(token, "agent1", "your-username", "your-repo")
    
    # Register this agent with its capabilities
    mcp.register_agent(
        capabilities=["code_review", "documentation", "bug_fixing"],
        metadata={"language_preferences": ["python", "javascript"], "version": "1.0.0"}
    )
    
    # Create a task for another agent
    task_id = mcp.create_task(
        task_type="code_review",
        parameters={
            "file_path": "src/main.py",
            "focus_areas": ["performance", "security"]
        },
        assigned_to="agent2",
        priority=2
    )
    
    # Send a message to another agent
    mcp.send_message(
        recipient_id="agent2",
        message_type="notification",
        content={
            "subject": "Code review requested",
            "message": f"I've assigned you a code review task: {task_id}"
        }
    )
    
    # Process incoming messages
    messages = mcp.get_messages(unread_only=True)
    message_ids = []
    
    for message in messages:
        print(f"Received message from {message['sender']}: {message['content']}")
        message_ids.append(message["id"])
        
        # Example: respond to task assignment
        if message["type"] == "task_assignment":
            task_id = message["content"]["task_id"]
            
            # Get task details
            tasks = mcp.get_tasks()
            task = next((t for t in tasks if t["id"] == task_id), None)
            
            if task:
                # Update task status to indicate we're working on it
                mcp.update_task_status(task_id, "pending")
                
                # Respond to the sender
                mcp.send_message(
                    recipient_id=message["sender"],
                    message_type="task_acknowledgment",
                    content={
                        "task_id": task_id,
                        "message": "I've started working on this task."
                    }
                )
    
    # Mark processed messages as read
    if message_ids:
        mcp.mark_messages_read(message_ids)
    
    # Share some data with other agents
    mcp.set_shared_data("latest_build_status", {
        "timestamp": mcp._get_timestamp(),
        "status": "success",
        "metrics": {
            "test_coverage": 87.2,
            "performance_score": 92
        }
    })
