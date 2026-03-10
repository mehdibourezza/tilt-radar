import asyncio
import sys
sys.path.insert(0, ".")

from data.db.session import create_all_tables

async def main():
    await create_all_tables()
    print("All tables created successfully.")

asyncio.run(main())
