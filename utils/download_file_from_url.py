import httpx


async def download_file_from_url(url: str) -> bytes:
    """Helper to download file bytes from a URL."""
    async with httpx.AsyncClient() as client:
        response = await client.get(url)
        response.raise_for_status()
        return response.content