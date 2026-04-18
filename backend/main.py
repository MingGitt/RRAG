import uvicorn


if __name__ == "__main__":
    uvicorn.run("api_server:app", host="127.0.0.1", port=5173, reload=True)