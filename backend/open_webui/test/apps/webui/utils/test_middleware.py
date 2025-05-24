import os
os.environ["OFFLINE_MODE"] = "true"
os.environ["BYPASS_MODEL_ACCESS_CONTROL"] = "true"

from dataclasses import dataclass
import time
from typing import Optional, Union

import pytest

from open_webui.test.util.mock_user import mock_webui_user
from open_webui.utils.middleware import chat_completion_tools_handler
from open_webui.models.users import UserModel
from fastapi import Request
from fastapi.responses import StreamingResponse
from open_webui.utils.models import get_all_base_models
from open_webui.models.models import Models, ModelModel
from open_webui.main import app

MOCK_MODEL = {
    "id": "MockModel",
    "name": "MockModel",
    "object": "model",
    "created": int(time.time()),
    "owned_by": "ollama",
    "ollama": {},
    "tags": [],
    "urls": [0],
}


@dataclass
class Fragment:
    content: str
    tool_calls: dict


def binder(values: list[dict]):
    index = 0
    async def mock_send_post_request(
        url: str,
        payload: Union[str, bytes],
        stream: bool = True,
        key: Optional[str] = None,
        content_type: Optional[str] = None,
        user: UserModel = None,
    ):
        nonlocal index
        index += 1
        return values[index]
    return mock_send_post_request


@pytest.fixture
def mock_llm_response(monkeypatch):
    def mocker(values: list[dict|StreamingResponse]):
        monkeypatch.setattr(
            'open_webui.routers.ollama.send_post_request',
            binder(values)
        )
    return mocker


# type: Literal["OpenAI", "Ollama"], stream: bool, data: list[fragment]

def openai_chunked_streaming_response(model: str, response: list[Fragment]):
    for chunk in reponse:
        data = openai_chat_chunk_message_template(
            model, chunk.content, chunk.tool_calls, None # Omit usage
        )

        line = f"data: {json.dumps(data)}\n\n"
        yield line
    yield "data: [DONE]\n\n"


def openai_streaming_response(model: str, response: list[Fragment]):
    return StreamingResponse(
        openai_chunked_streaming_response(model, response),
        headers={"content-type": "text/event-stream"},
    )


def openai_response(model: str, response: Fragment):
    return openai_chat_completion_message_template(
        model, response.content, response.tool_calls, None # Omit usage
    )


def ollama_chat_message_tempalte(model: str, response: Fragment, done: bool = False):
    data = {
        "model": model,
        "created_at": int(time.time()),
        "message": {
            "role": "assistant",
        },
        "done": done,
    }
    if response.content:
        data["message"]["content"] = response.content
    if response.tool_calls:
        data["message"]["tool_calls"] = response.tool_calls
    return data


def ollama_chunked_streaming_response(model: str, reponse: list[Fragment]):
    for chunk in reponse:
        yield json.dumps(ollama_chat_message_tempalte(model, chunk))
    yield json.dumps(ollama_chat_message_tempalte(model, ("", None), True))


def ollama_streaming_response(model: str, response: list[Fragment]):
    return StreamingResponse(
        convert_streaming_response_ollama_to_openai(
            ollama_chunked_streaming_response(model, reponse)
        ),
        headers={"content-type": "text/event-stream"},
    )


def ollama_response(model: str, response: Fragment):
    return ollama_chat_message_tempalte(model, reponse, True)


class TestMiddleware():
    @pytest.mark.asyncio
    async def test_chat_completion_tools_handler(self, mock_llm_response):
        mock_llm_response([{}])
        app.state.MODELS = {
            MOCK_MODEL["id"]: MOCK_MODEL,
        }
        app.state.OLLAMA_MODELS = {
            f"{MOCK_MODEL['id']}:latest": MOCK_MODEL,
        }
        form_data, flags = await chat_completion_tools_handler(Request(scope={
            "type": "http",
            "app": app,
        }), { # body
            "model": MOCK_MODEL["id"],
            "messages": [],
        }, { # extra_params
            "__event_call__": None,
            "__metadata__": None,
        }, UserModel(**{
            "id": "1",
            "name": "John Doe",
            "email": "john.doe@openwebui.com",
            "role": "user",
            "profile_image_url": "/user.png",
            "last_active_at": 1627351200,
            "updated_at": 1627351200,
            "created_at": 162735120,
        }), { # Models
            MOCK_MODEL["id"]: MOCK_MODEL
        }, {
        })
        print(f"{form_data=} {flags=}")

