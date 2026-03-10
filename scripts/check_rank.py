import asyncio
import sys
sys.path.insert(0, ".")
from data.riot.client import RiotClient

async def main():
    async with RiotClient() as riot:
        puuid = await riot.get_puuid("MarreDesNoobsNul", "007")
        rank = await riot.get_rank(puuid)
        if rank:
            print(f"Tier: {rank['tier']} {rank['rank']}")
            print(f"LP: {rank['leaguePoints']}")
            print(f"W/L: {rank['wins']}W {rank['losses']}L")
        else:
            print("Unranked")

asyncio.run(main())
