import redis
import os
from dotenv import load_dotenv
from loguru import logger

load_dotenv()

redis_client = None


def redis_init():

    """Function that intializes redis for caching data"""

    global redis_client
    
    logger.info("Initializing Redis...")

    REDIS_HOST = os.getenv("REDIS_HOST")
    REDIS_PASSWORD = os.getenv("REDIS_PASSWORD")

    if not REDIS_HOST:
        logger.warning("Redis host not found")
        return 
    if not REDIS_PASSWORD:
        logger.warning("Redis password not found")
        return 

    redis_client = redis.Redis(
        host= REDIS_HOST,
        port=15457,
        decode_responses=True,
        username="default",
        password=REDIS_PASSWORD,
    )

