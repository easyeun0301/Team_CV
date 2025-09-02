import asyncio

async def run_server(host: str = "0.0.0.0", port: int = 8080):
    print(f"[server] starting aiortc server on {host}:{port}")
    # TODO: add aiohttp signaling + aiortc PeerConnection here

if __name__ == "__main__":
    asyncio.run(run_server())
