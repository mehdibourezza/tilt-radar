import asyncio
import asyncpg
import redis

async def test_postgres():
    try:
        conn = await asyncpg.connect("postgresql://postgres:postgres@localhost:5432/tiltRadar")
        await conn.close()
        print("PostgreSQL OK")
    except Exception as e:
        print(f"PostgreSQL FAILED: {e}")

def test_redis():
    try:
        r = redis.Redis(host="localhost", port=6379)
        r.ping()
        print("Redis OK")
    except Exception as e:
        print(f"Redis FAILED: {e}")

asyncio.run(test_postgres())
test_redis()
