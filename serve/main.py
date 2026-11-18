import os
os.environ.setdefault("PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION", "python")

from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from api.routers import analysis, training, inference
from configs.config import GLOBAL_LOGGER
from utils.nacos import build_registrar_from_env

@asynccontextmanager
async def lifespan(app: FastAPI):
    # Nacos registration is optional and controlled by env var NACOS_ENABLED.
    registrar = None
    try:
        registrar = build_registrar_from_env(default_port=int(os.getenv("PORT", "8000")))
        if registrar is not None:
            await registrar.start()
            spec = registrar.spec
            GLOBAL_LOGGER.info(
                f"Nacos registered: service={spec.service_name} ip={spec.ip} port={spec.port} "
                f"group={spec.group_name} namespace={spec.namespace_id} ephemeral={spec.ephemeral}"
            )
    except Exception as e:
        GLOBAL_LOGGER.exception(f"Nacos registration failed: {e}")

    yield

    if registrar is not None:
        try:
            await registrar.close()
            GLOBAL_LOGGER.info("Nacos deregistered")
        except Exception as e:
            GLOBAL_LOGGER.exception(f"Nacos deregistration failed: {e}")


app = FastAPI(title="AutoApex TimeSeries Analysis Platform", lifespan=lifespan)

# CORS 配置
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 注册路由
app.include_router(analysis.router)
app.include_router(training.router)
app.include_router(inference.router)

if __name__ == "__main__":
    import uvicorn
    port = int(os.getenv("PORT", "8000"))
    uvicorn.run(app, host="0.0.0.0", port=port)