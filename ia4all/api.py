from fastapi import FastAPI
from fastapi.responses import JSONResponse
from fastapi.exceptions import RequestValidationError
from fastapi.exceptions import ValidationError
from starlette.requests import Request
from starlette.exceptions import HTTPException
from django.core.handlers.asgi import ASGIHandler

app = FastAPI()

@app.get("/hello")
def read_root():
    return {"Hello": "World"}

@app.get("/items/{item_id}")
def read_item(item_id: int, q: str = None):
    return {"item_id": item_id, "q": q}

@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    raise ValidationError(exc.errors(), exc)

@app.exception_handler(HTTPException)
async def http_exception_handler(request: Request, exc: HTTPException):
    return await ASGIHandler().get_response(request.scope)(request.scope, request.receive, request.send)
