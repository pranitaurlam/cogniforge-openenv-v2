from fastapi import FastAPI
from .env import SupportEnv, Action, Observation, Reward
from .tasks import TASKS_DATA

app = FastAPI(title="Support Agent OpenEnv")
_env = SupportEnv(TASKS_DATA)


@app.get("/")
async def root():
    return {"status": "ok", "message": "Support Agent OpenEnv is running."}


@app.post("/reset", response_model=Observation)
async def reset():
    return _env.reset()


@app.post("/step")
async def step(action: Action):
    obs, reward, done, info = _env.step(action)
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
async def state():
    return _env.state()


def main():
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=7860)


if __name__ == "__main__":
    main()
