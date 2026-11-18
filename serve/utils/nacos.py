from __future__ import annotations

import asyncio
import json
import os
import socket
from dataclasses import dataclass
from typing import Any, Optional
from urllib.parse import urlparse

import httpx


def _first_nacos_base_url(server_addresses: str) -> str:
    """
    Accepts formats like:
    - "http://127.0.0.1:8848"
    - "127.0.0.1:8848"
    - "http://a:8848,http://b:8848"
    """
    first = server_addresses.split(",")[0].strip()
    if not first:
        raise ValueError("NACOS_SERVER_ADDRESSES is empty")
    if "://" not in first:
        first = "http://" + first
    parsed = urlparse(first)
    if not parsed.scheme or not parsed.netloc:
        raise ValueError(f"Invalid NACOS_SERVER_ADDRESSES: {server_addresses!r}")
    return f"{parsed.scheme}://{parsed.netloc}"


def _guess_local_ip(target_host: str) -> str:
    """
    Best-effort local IP detection by creating a UDP socket and 'connecting'
    to the target host. This does not send packets but lets OS pick the
    correct outbound interface.
    """
    s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    try:
        s.connect((target_host, 80))
        ip = s.getsockname()[0]
    finally:
        s.close()
    return ip


@dataclass(frozen=True)
class NacosInstanceSpec:
    server_addresses: str
    service_name: str
    ip: str
    port: int
    namespace_id: str = "public"
    group_name: str = "DEFAULT_GROUP"
    cluster_name: str = "DEFAULT"
    weight: float = 1.0
    enabled: bool = True
    healthy: bool = True
    ephemeral: bool = True
    metadata: Optional[dict[str, Any]] = None


class NacosRegistrar:
    """
    Minimal Nacos Naming registration via OpenAPI (HTTP).

    OpenAPI (v1) endpoints used:
    - Register instance:   POST   /nacos/v1/ns/instance
    - Deregister instance: DELETE /nacos/v1/ns/instance
    - Beat (ephemeral):    PUT    /nacos/v1/ns/instance/beat
    """

    def __init__(self, spec: NacosInstanceSpec, *, timeout_s: float = 5.0, beat_interval_s: float = 5.0):
        self._spec = spec
        self._timeout_s = timeout_s
        self._beat_interval_s = beat_interval_s
        self._client: Optional[httpx.AsyncClient] = None
        self._beat_task: Optional[asyncio.Task[None]] = None
        self._username: Optional[str] = None
        self._password: Optional[str] = None
        self._access_token: Optional[str] = None

    @property
    def spec(self) -> NacosInstanceSpec:
        return self._spec

    def with_auth(self, *, username: Optional[str], password: Optional[str]) -> "NacosRegistrar":
        self._username = (username or "").strip() or None
        self._password = (password or "").strip() or None
        return self

    async def start(self) -> None:
        base_url = _first_nacos_base_url(self._spec.server_addresses)
        self._client = httpx.AsyncClient(base_url=base_url, timeout=self._timeout_s)

        if self._username and self._password:
            await self._login_and_set_token()

        # Register instance (creates service implicitly if missing)
        await self._register_instance()

        # Start heartbeat for ephemeral instance
        if self._spec.ephemeral:
            self._beat_task = asyncio.create_task(self._beat_loop(), name="nacos-beat")

    async def close(self) -> None:
        # Stop beat loop first (avoid sending beat after deregister)
        if self._beat_task is not None:
            self._beat_task.cancel()
            try:
                await self._beat_task
            except asyncio.CancelledError:
                pass
            self._beat_task = None

        # Deregister instance
        try:
            await self._deregister_instance()
        finally:
            if self._client is not None:
                await self._client.aclose()
                self._client = None

    async def _register_instance(self) -> None:
        assert self._client is not None
        params: dict[str, Any] = {
            "serviceName": self._spec.service_name,
            "ip": self._spec.ip,
            "port": self._spec.port,
            "namespaceId": self._spec.namespace_id,
            "groupName": self._spec.group_name,
            "clusterName": self._spec.cluster_name,
            "weight": str(self._spec.weight),
            "enabled": str(self._spec.enabled).lower(),
            "healthy": str(self._spec.healthy).lower(),
            "ephemeral": str(self._spec.ephemeral).lower(),
        }
        if self._access_token:
            params["accessToken"] = self._access_token
        if self._spec.metadata:
            params["metadata"] = json.dumps(self._spec.metadata, ensure_ascii=False)

        resp = await self._client.post("/nacos/v1/ns/instance", params=params)
        if resp.status_code == 403 and self._username and self._password:
            await self._login_and_set_token()
            params["accessToken"] = self._access_token
            resp = await self._client.post("/nacos/v1/ns/instance", params=params)
        resp.raise_for_status()
        # Nacos returns "ok" (text) on success.

    async def _deregister_instance(self) -> None:
        assert self._client is not None
        params: dict[str, Any] = {
            "serviceName": self._spec.service_name,
            "ip": self._spec.ip,
            "port": self._spec.port,
            "namespaceId": self._spec.namespace_id,
            "groupName": self._spec.group_name,
            "clusterName": self._spec.cluster_name,
            "ephemeral": str(self._spec.ephemeral).lower(),
        }
        if self._access_token:
            params["accessToken"] = self._access_token
        resp = await self._client.delete("/nacos/v1/ns/instance", params=params)
        # Some deployments may return 404 if already removed; treat as ok.
        if resp.status_code == 403 and self._username and self._password:
            await self._login_and_set_token()
            params["accessToken"] = self._access_token
            resp = await self._client.delete("/nacos/v1/ns/instance", params=params)
        if resp.status_code not in (200, 404):
            resp.raise_for_status()

    async def _beat_loop(self) -> None:
        assert self._client is not None
        while True:
            try:
                await self._send_beat()
            except Exception:
                # Heartbeat failures should not crash the whole app; retry.
                pass
            await asyncio.sleep(self._beat_interval_s)

    async def _send_beat(self) -> None:
        assert self._client is not None
        beat = {
            "serviceName": self._spec.service_name,
            "ip": self._spec.ip,
            "port": self._spec.port,
            "cluster": self._spec.cluster_name,
            "weight": self._spec.weight,
            "metadata": self._spec.metadata or {},
        }
        params: dict[str, Any] = {
            "serviceName": self._spec.service_name,
            "ip": self._spec.ip,
            "port": self._spec.port,
            "namespaceId": self._spec.namespace_id,
            "groupName": self._spec.group_name,
            "ephemeral": str(self._spec.ephemeral).lower(),
            "beat": json.dumps(beat, ensure_ascii=False),
        }
        if self._access_token:
            params["accessToken"] = self._access_token
        resp = await self._client.put("/nacos/v1/ns/instance/beat", params=params)
        if resp.status_code == 403 and self._username and self._password:
            await self._login_and_set_token()
            params["accessToken"] = self._access_token
            resp = await self._client.put("/nacos/v1/ns/instance/beat", params=params)
        resp.raise_for_status()

    async def _login_and_set_token(self) -> None:
        """
        Nacos OpenAPI auth requires calling login first, then append accessToken to URL params.
        Common paths seen across versions:
        - POST /nacos/v1/auth/login
        - POST /nacos/v1/auth/users/login
        - POST /nacos/v3/auth/user/login
        """
        assert self._client is not None
        assert self._username and self._password

        candidates = (
            "/nacos/v1/auth/login",
            "/nacos/v1/auth/users/login",
            "/nacos/v3/auth/user/login",
        )
        last_err: Optional[Exception] = None
        for path in candidates:
            try:
                resp = await self._client.post(
                    path,
                    data={"username": self._username, "password": self._password},
                    headers={"Content-Type": "application/x-www-form-urlencoded"},
                )
                resp.raise_for_status()
                data = resp.json()
                token = data.get("accessToken") if isinstance(data, dict) else None
                if not token:
                    raise ValueError(f"Login response missing accessToken: {data!r}")
                self._access_token = str(token)
                return
            except Exception as e:
                last_err = e
                continue
        raise RuntimeError(f"Failed to login to Nacos via OpenAPI: {last_err}") from last_err


def build_registrar_from_env(*, default_port: int) -> Optional[NacosRegistrar]:
    """
    Env vars (aligned with your document naming as much as possible):
    - NACOS_ENABLED: "true"/"false" (default false)
    - NACOS_SERVER_ADDRESSES: e.g. "http://172.19.2.191:8848"
    - NACOS_SERVICE_NAME: service name in nacos
    - NACOS_NAMESPACE_ID: default "public"
    - NACOS_GROUP_NAME: default "DEFAULT_GROUP"
    - NACOS_CLUSTER_NAME: default "DEFAULT"
    - NACOS_INSTANCE_IP: optional, otherwise auto-detect
    - NACOS_INSTANCE_PORT: optional, otherwise default_port
    - NACOS_EPHEMERAL: default true
    - NACOS_METADATA_JSON: optional JSON string
    - NACOS_BEAT_INTERVAL_S: default 5
    """
    enabled = os.getenv("NACOS_ENABLED", "").strip().lower() in ("1", "true", "yes", "on")
    if not enabled:
        return None

    server_addresses = (os.getenv("NACOS_SERVER_ADDRESSES") or "").strip()
    service_name = (os.getenv("NACOS_SERVICE_NAME") or "").strip()
    if not server_addresses:
        raise ValueError("NACOS_ENABLED=true but NACOS_SERVER_ADDRESSES is missing")
    if not service_name:
        raise ValueError("NACOS_ENABLED=true but NACOS_SERVICE_NAME is missing")

    base_url = _first_nacos_base_url(server_addresses)
    target_host = urlparse(base_url).hostname or "127.0.0.1"

    ip = (os.getenv("NACOS_INSTANCE_IP") or "").strip() or _guess_local_ip(target_host)
    port = int((os.getenv("NACOS_INSTANCE_PORT") or str(default_port)).strip())

    namespace_id = (os.getenv("NACOS_NAMESPACE_ID") or "public").strip()
    group_name = (os.getenv("NACOS_GROUP_NAME") or "DEFAULT_GROUP").strip()
    cluster_name = (os.getenv("NACOS_CLUSTER_NAME") or "DEFAULT").strip()
    ephemeral = (os.getenv("NACOS_EPHEMERAL") or "true").strip().lower() in ("1", "true", "yes", "on")

    metadata_json = (os.getenv("NACOS_METADATA_JSON") or "").strip()
    metadata: Optional[dict[str, Any]] = None
    if metadata_json:
        metadata = json.loads(metadata_json)
        if not isinstance(metadata, dict):
            raise ValueError("NACOS_METADATA_JSON must be a JSON object")

    beat_interval_s = float((os.getenv("NACOS_BEAT_INTERVAL_S") or "5").strip())

    spec = NacosInstanceSpec(
        server_addresses=server_addresses,
        service_name=service_name,
        ip=ip,
        port=port,
        namespace_id=namespace_id,
        group_name=group_name,
        cluster_name=cluster_name,
        ephemeral=ephemeral,
        metadata=metadata,
    )
    registrar = NacosRegistrar(spec, beat_interval_s=beat_interval_s)
    registrar.with_auth(
        username=os.getenv("NACOS_USERNAME"),
        password=os.getenv("NACOS_PASSWORD"),
    )
    return registrar

