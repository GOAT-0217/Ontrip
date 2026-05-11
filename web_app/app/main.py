import sys
import os
from dotenv import load_dotenv

load_dotenv()

sys.path.append(os.path.join(os.path.dirname(__file__), "..", ".."))

from fastapi import FastAPI, Request, Depends, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from pydantic import BaseModel
from starlette.middleware.sessions import SessionMiddleware
import uuid

from customer_support_chat.app.services.chat_service import process_user_message
from .core.user_data_manager import (
    get_user_session,
    update_user_chat_history,
    get_pending_action,
    set_user_decision,
    clear_pending_action,
    clear_user_decision,
    get_operation_log
)
from .core.auth import authenticate_user, create_user, get_user_by_id

# Load environment variables
load_dotenv()

app = FastAPI()

app.add_middleware(
    SessionMiddleware,
    secret_key=os.environ.get("SECRET_KEY", "fallback-dev-key-change-me"),
    session_cookie="ontrip_session",
    max_age=86400,
)

# Mount static files directory
app.mount("/static", StaticFiles(directory=os.path.join(os.path.dirname(__file__), "static")), name="static")

# Set up Jinja2 templates
templates = Jinja2Templates(directory=os.path.join(os.path.dirname(__file__), "templates"))

class ChatMessage(BaseModel):
    message: str

class ApprovalDecision(BaseModel):
    decision: str

async def require_auth(request: Request):
    """Require valid user session for API endpoints."""
    user = request.session.get("user")
    if not user:
        raise HTTPException(status_code=401, detail="Authentication required")
    db_user = get_user_by_id(user["id"])
    if not db_user:
        request.session.clear()
        raise HTTPException(status_code=401, detail="User no longer exists")
    return db_user


def get_session_data(request: Request):
    """Get or create session data for the current user."""
    session_id = request.cookies.get("session_id")
    if not session_id:
        session_id = str(uuid.uuid4())
    
    # Get the user session data
    session_data = get_user_session(session_id)
    
    # Ensure config exists in session_data
    if "config" not in session_data:
        session_data["config"] = {
            "thread_id": session_id,
            "passenger_id": "5102 899977"  # Default passenger ID
        }
    
    return {
        "session_id": session_id,
        "config": session_data["config"],
        "user_data": session_data
    }

@app.get("/", response_class=HTMLResponse)
async def get_chat_page(request: Request, session_data: dict = Depends(get_session_data)):
    """Serve the chat interface. Requires authentication."""
    user = request.session.get("user")
    if not user:
        return RedirectResponse(url="/login", status_code=303)

    response = templates.TemplateResponse(request, "chat.html", {
        "session_id": session_data["session_id"],
        "chat_history": session_data["user_data"].get("chat_history", []),
        "username": user["username"],
    })
    response.set_cookie(key="session_id", value=session_data["session_id"])
    return response

@app.post("/chat")
async def chat(chat_message: ChatMessage, user: dict = Depends(require_auth), session_data: dict = Depends(get_session_data)):
    """Process a chat message and return the AI response."""
    try:
        # Process the user message
        ai_response = await process_user_message(session_data, chat_message.message)
        
        # Update the user's chat history
        update_user_chat_history(session_data["session_id"], chat_message.message, ai_response)
        
        # Return the AI response
        return JSONResponse(content={"response": ai_response})
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing chat message: {e}")
        # Return a user-friendly error message
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)

@app.post("/new-chat")
async def new_chat(request: Request, user: dict = Depends(require_auth)):
    """Start a new conversation by generating a new session ID."""
    from .core.user_data_manager import clear_session_data
    
    new_session_id = str(uuid.uuid4())
    
    # Clear old session data if exists
    old_session_id = request.cookies.get("session_id")
    if old_session_id:
        clear_session_data(old_session_id)
    
    response = JSONResponse(content={"session_id": new_session_id})
    response.set_cookie(key="session_id", value=new_session_id)
    return response

# HITL (Human-in-the-Loop) endpoints

@app.get("/pending-action")
async def get_pending_action_endpoint(user: dict = Depends(require_auth), session_data: dict = Depends(get_session_data)):
    """Check if there is a pending action requiring user approval."""
    try:
        pending_action = get_pending_action(session_data["session_id"])
        if pending_action:
            return JSONResponse(content={"pending_action": pending_action})
        else:
            return JSONResponse(content={"pending_action": None})
    except Exception as e:
        print(f"Error checking pending action: {e}")
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)


@app.post("/approve-action")
async def approve_action(request: Request, user: dict = Depends(require_auth), session_data: dict = Depends(get_session_data)):
    """Approve a pending action."""
    try:
        # Process the user's approval decision
        from customer_support_chat.app.services.chat_service import process_user_decision
        ai_response = await process_user_decision(session_data, "approve")
        
        # Update the user's chat history
        update_user_chat_history(session_data["session_id"], "[User approved action]", ai_response)
        
        # Return the AI response
        return JSONResponse(content={"response": ai_response})
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing approval: {e}")
        # Return a user-friendly error message
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)


@app.post("/reject-action")
async def reject_action(request: Request, user: dict = Depends(require_auth), session_data: dict = Depends(get_session_data)):
    """Reject a pending action."""
    try:
        # Process the user's rejection decision
        from customer_support_chat.app.services.chat_service import process_user_decision
        ai_response = await process_user_decision(session_data, "reject")
        
        # Update the user's chat history
        update_user_chat_history(session_data["session_id"], "[User rejected action]", ai_response)
        
        # Return the AI response
        return JSONResponse(content={"response": ai_response})
        
    except Exception as e:
        # Log the error for debugging
        print(f"Error processing rejection: {e}")
        # Return a user-friendly error message
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)

@app.get("/operation-log")
async def get_operation_log_endpoint(user: dict = Depends(require_auth), session_data: dict = Depends(get_session_data)):
    """Get the operation log for the current session."""
    try:
        # Get only the most recent 20 log entries to reduce data transfer
        operation_log = get_operation_log(session_data["session_id"], limit=20)
        return JSONResponse(content={"operation_log": operation_log})
    except Exception as e:
        print(f"Error retrieving operation log: {e}")
        return JSONResponse(content={"error": "An unexpected error occurred. Please try again later."}, status_code=500)


# ── Authentication routes ──────────────────────────────────────────


@app.get("/login", response_class=HTMLResponse)
async def get_login_page(request: Request):
    """Serve the login page. Redirect to chat if already authenticated."""
    if request.session.get("user"):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "login.html", {"error": None})


@app.post("/login")
async def login(request: Request):
    """Process login form submission."""
    form_data = await request.form()
    username = form_data.get("username", "").strip()
    password = form_data.get("password", "")

    if not username or not password:
        return templates.TemplateResponse(request, "login.html", {
            "error": "请输入用户名和密码。"
        }, status_code=400)

    user = authenticate_user(username, password)
    if not user:
        return templates.TemplateResponse(request, "login.html", {
            "error": "用户名或密码错误。"
        }, status_code=401)

    request.session["user"] = {"id": user["id"], "username": user["username"]}
    return RedirectResponse(url="/", status_code=303)


@app.get("/register", response_class=HTMLResponse)
async def get_register_page(request: Request):
    """Serve the registration page. Redirect to chat if already authenticated."""
    if request.session.get("user"):
        return RedirectResponse(url="/", status_code=303)
    return templates.TemplateResponse(request, "register.html", {"error": None})


@app.post("/register")
async def register(request: Request):
    """Process registration form submission."""
    form_data = await request.form()
    username = form_data.get("username", "").strip()
    password = form_data.get("password", "")
    confirm = form_data.get("confirm_password", "")

    if not username or not password:
        return templates.TemplateResponse(request, "register.html", {
            "error": "请填写所有字段。"
        }, status_code=400)

    if len(username) < 3 or len(username) > 50:
        return templates.TemplateResponse(request, "register.html", {
            "error": "用户名长度需在 3 到 50 个字符之间。"
        }, status_code=400)

    if len(password) < 6:
        return templates.TemplateResponse(request, "register.html", {
            "error": "密码长度不能少于 6 个字符。"
        }, status_code=400)

    if password != confirm:
        return templates.TemplateResponse(request, "register.html", {
            "error": "两次输入的密码不一致。"
        }, status_code=400)

    user_id = create_user(username, password)
    if user_id is None:
        return templates.TemplateResponse(request, "register.html", {
            "error": "用户名已被占用，请换一个。"
        }, status_code=409)

    request.session["user"] = {"id": user_id, "username": username}
    return RedirectResponse(url="/", status_code=303)


@app.get("/logout")
async def logout(request: Request):
    """Clear user session and redirect to login page."""
    request.session.clear()
    return RedirectResponse(url="/login", status_code=303)