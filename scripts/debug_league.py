import asyncio
import sys
sys.path.insert(0, ".")
from data.riot.client import RiotClient

async def main():
    async with RiotClient() as riot:
        url = "https://euw1.api.riotgames.com/lol/league/v4/entries/RANKED_SOLO_5x5/SILVER/IV"
        data = await riot._request(url, params={"page": 1})
        print(f"Total entries returned: {len(data)}")
        if data:
            print(f"First entry keys: {list(data[0].keys())}")
            print(f"First entry: {data[0]}")

asyncio.run(main())
