import os
import datajoint as dj

if "custom" not in dj.config:
    dj.config["custom"] = {}

# overwrite dj.config['custom'] values with environment variables if available
dj.config["custom"]["database.prefix"] = os.getenv(
    "DATABASE_PREFIX", dj.config["custom"].get("database.prefix", "")
)

db_prefix = dj.config["custom"]["database.prefix"]

dj.config["stores"] = {
    "neurophotonics": dict(
        protocol="s3",
        endpoint="s3.amazonaws.com:9000",
        bucket="dj-sciops",
        location=f"{db_prefix[:-1]}/datajoint/blob",
        access_key=os.getenv("AWS_ACCESS_KEY", None),
        secret_key=os.getenv("AWS_ACCESS_SECRET", None),
        stage="/main/datajoint/blob",
    ),
}

from . import space, fields, design, sim, demix
