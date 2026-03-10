import asyncio
import sys
sys.path.insert(0, ".")

from data.riot.client import RiotClient

async def main(name, tag):
    async with RiotClient() as riot:
        puuid = await riot.get_puuid(name, tag)
        if puuid:
            print(f"Found: {name}#{tag}")
            print(f"PUUID: {puuid[:20]}...")
        else:
            print(f"Not found: {name}#{tag} — check your tag")

asyncio.run(main(sys.argv[1], sys.argv[2]))
