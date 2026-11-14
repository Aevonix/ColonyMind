"""HTTP checks with a total deadline, including response body consumption."""
import asyncio

import httpx


async def _get(url, timeout, key, body):
    headers = {'X-API-Key': key} if key else {}
    async with asyncio.timeout(timeout):
        async with httpx.AsyncClient(trust_env=False, follow_redirects=False) as client:
            async with client.stream('GET', url, headers=headers, timeout=timeout) as response:
                chunks = bytearray()
                async for chunk in response.aiter_bytes():
                    chunks.extend(chunk)
                    if len(chunks) > 4 * 1024 * 1024:
                        raise ValueError('response_too_large')
                payload = httpx.Response(200, content=bytes(chunks)).json() if body else None
                return response.status_code, payload


def fetch(url, timeout, *, key=None, body=False):
    return asyncio.run(_get(url, timeout, key, body))
