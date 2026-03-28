"""Database and streaming connectors — MySQL, PostgreSQL, MSSQL, Oracle,
MongoDB, Redis, Kafka, Elasticsearch, SQLite, S3, BigQuery, ClickHouse."""

from __future__ import annotations

import json
import csv as csv_mod
import io
from pathlib import Path
from typing import Any


CONNECTOR_REGISTRY: dict[str, dict] = {
    "mysql": {
        "name": "MySQL",
        "category": "sql",
        "params": [
            {"name": "host", "label": "Host", "default": "localhost"},
            {"name": "port", "label": "Port", "type": "int", "default": 3306},
            {"name": "user", "label": "User", "default": "root"},
            {"name": "password", "label": "Password", "secret": True},
            {"name": "database", "label": "Database"},
        ],
        "pip": "pymysql",
    },
    "postgresql": {
        "name": "PostgreSQL",
        "category": "sql",
        "params": [
            {"name": "host", "default": "localhost"},
            {"name": "port", "type": "int", "default": 5432},
            {"name": "user", "default": "postgres"},
            {"name": "password", "secret": True},
            {"name": "database"},
        ],
        "pip": "psycopg2-binary",
    },
    "mssql": {
        "name": "Microsoft SQL Server",
        "category": "sql",
        "params": [
            {"name": "host", "default": "localhost"},
            {"name": "port", "type": "int", "default": 1433},
            {"name": "user"},
            {"name": "password", "secret": True},
            {"name": "database"},
        ],
        "pip": "pymssql",
    },
    "oracle": {
        "name": "Oracle",
        "category": "sql",
        "params": [
            {"name": "host", "default": "localhost"},
            {"name": "port", "type": "int", "default": 1521},
            {"name": "user"},
            {"name": "password", "secret": True},
            {"name": "service_name"},
        ],
        "pip": "oracledb",
    },
    "sqlite": {
        "name": "SQLite",
        "category": "sql",
        "params": [
            {"name": "path", "label": "Database File Path", "default": "./data.db"},
        ],
        "pip": None,
    },
    "mongodb": {
        "name": "MongoDB",
        "category": "nosql",
        "params": [
            {"name": "uri", "label": "Connection URI", "default": "mongodb://localhost:27017"},
            {"name": "database"},
            {"name": "collection"},
        ],
        "pip": "pymongo",
    },
    "redis": {
        "name": "Redis",
        "category": "nosql",
        "params": [
            {"name": "host", "default": "localhost"},
            {"name": "port", "type": "int", "default": 6379},
            {"name": "password", "secret": True, "default": ""},
            {"name": "db", "type": "int", "default": 0},
        ],
        "pip": "redis",
    },
    "elasticsearch": {
        "name": "Elasticsearch",
        "category": "nosql",
        "params": [
            {"name": "hosts", "label": "Host(s)", "default": "http://localhost:9200"},
            {"name": "index", "label": "Index Name"},
            {"name": "api_key", "secret": True, "default": ""},
        ],
        "pip": "elasticsearch",
    },
    "kafka": {
        "name": "Apache Kafka",
        "category": "streaming",
        "params": [
            {"name": "bootstrap_servers", "default": "localhost:9092"},
            {"name": "topic"},
            {"name": "group_id", "default": "sg_consumer"},
            {"name": "auto_offset_reset", "default": "earliest"},
        ],
        "pip": "kafka-python",
    },
    "s3": {
        "name": "AWS S3 / MinIO",
        "category": "cloud",
        "params": [
            {"name": "endpoint_url", "default": ""},
            {"name": "bucket"},
            {"name": "prefix", "default": ""},
            {"name": "aws_access_key_id", "secret": True},
            {"name": "aws_secret_access_key", "secret": True},
            {"name": "region", "default": "us-east-1"},
        ],
        "pip": "boto3",
    },
    "bigquery": {
        "name": "Google BigQuery",
        "category": "cloud",
        "params": [
            {"name": "project"},
            {"name": "dataset"},
            {"name": "credentials_json", "label": "Service Account JSON path"},
        ],
        "pip": "google-cloud-bigquery",
    },
    "clickhouse": {
        "name": "ClickHouse",
        "category": "sql",
        "params": [
            {"name": "host", "default": "localhost"},
            {"name": "port", "type": "int", "default": 9000},
            {"name": "user", "default": "default"},
            {"name": "password", "secret": True, "default": ""},
            {"name": "database", "default": "default"},
        ],
        "pip": "clickhouse-connect",
    },
    "csv_file": {
        "name": "CSV File",
        "category": "file",
        "params": [
            {"name": "path", "label": "File Path"},
            {"name": "delimiter", "default": ","},
            {"name": "encoding", "default": "utf-8"},
        ],
        "pip": None,
    },
    "json_file": {
        "name": "JSON / JSONL File",
        "category": "file",
        "params": [
            {"name": "path", "label": "File Path"},
            {"name": "lines", "type": "bool", "default": False, "label": "JSONL (one object per line)"},
        ],
        "pip": None,
    },
    "parquet": {
        "name": "Parquet File",
        "category": "file",
        "params": [
            {"name": "path", "label": "File Path"},
        ],
        "pip": "pyarrow",
    },
}


class DataConnector:
    """Unified interface for all data sources and sinks."""

    def __init__(self, connector_type: str, params: dict[str, Any]):
        if connector_type not in CONNECTOR_REGISTRY:
            raise ValueError(f"Unknown connector: {connector_type}")
        self.type = connector_type
        self.config = CONNECTOR_REGISTRY[connector_type]
        self.params = params
        self._conn = None

    def test_connection(self) -> dict:
        """Test if the connection works."""
        try:
            self._connect()
            self._disconnect()
            return {"status": "ok", "message": "Connection successful"}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def load(self, query: str | None = None, limit: int = 10000) -> dict:
        """Load data from the source. Returns rows + column info."""
        try:
            self._connect()
            rows, columns = self._load(query, limit)
            self._disconnect()
            return {
                "status": "ok",
                "rows": rows,
                "columns": columns,
                "count": len(rows),
            }
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def preview(self, query: str | None = None, limit: int = 20) -> dict:
        return self.load(query, limit)

    def sink(self, rows: list[dict], target: str | None = None) -> dict:
        """Write data to the destination."""
        try:
            self._connect()
            count = self._sink(rows, target)
            self._disconnect()
            return {"status": "ok", "written": count}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    def list_tables(self) -> dict:
        """List available tables/collections/topics."""
        try:
            self._connect()
            tables = self._list_tables()
            self._disconnect()
            return {"status": "ok", "tables": tables}
        except Exception as e:
            return {"status": "error", "message": str(e)}

    # ── Internal dispatch ──

    def _connect(self):
        cat = self.config["category"]
        if cat == "sql":
            self._connect_sql()
        elif self.type == "mongodb":
            self._connect_mongo()
        elif self.type == "redis":
            self._connect_redis()
        elif self.type == "elasticsearch":
            self._connect_es()
        elif self.type == "kafka":
            self._connect_kafka()
        elif self.type == "s3":
            self._connect_s3()
        elif cat == "file":
            pass  # No connection needed

    def _disconnect(self):
        if self._conn:
            try:
                self._conn.close()
            except Exception:
                pass
            self._conn = None

    # ── SQL (MySQL, PostgreSQL, MSSQL, Oracle, SQLite, ClickHouse) ──

    def _connect_sql(self):
        p = self.params
        if self.type == "mysql":
            import pymysql
            self._conn = pymysql.connect(host=p["host"], port=int(p.get("port", 3306)),
                user=p["user"], password=p.get("password", ""), database=p["database"],
                cursorclass=pymysql.cursors.DictCursor)
        elif self.type == "postgresql":
            import psycopg2
            import psycopg2.extras
            self._conn = psycopg2.connect(host=p["host"], port=int(p.get("port", 5432)),
                user=p["user"], password=p.get("password", ""), dbname=p["database"])
        elif self.type == "mssql":
            import pymssql
            self._conn = pymssql.connect(server=p["host"], port=int(p.get("port", 1433)),
                user=p["user"], password=p.get("password", ""), database=p["database"])
        elif self.type == "oracle":
            import oracledb
            self._conn = oracledb.connect(user=p["user"], password=p.get("password", ""),
                dsn=f"{p['host']}:{p.get('port', 1521)}/{p['service_name']}")
        elif self.type == "sqlite":
            import sqlite3
            self._conn = sqlite3.connect(p["path"])
            self._conn.row_factory = sqlite3.Row
        elif self.type == "clickhouse":
            import clickhouse_connect
            self._conn = clickhouse_connect.get_client(host=p["host"], port=int(p.get("port", 8123)),
                username=p.get("user", "default"), password=p.get("password", ""), database=p.get("database", "default"))

    def _load(self, query, limit):
        cat = self.config["category"]
        if cat == "sql":
            return self._load_sql(query, limit)
        elif self.type == "mongodb":
            return self._load_mongo(query, limit)
        elif self.type == "redis":
            return self._load_redis(query, limit)
        elif self.type == "elasticsearch":
            return self._load_es(query, limit)
        elif self.type == "kafka":
            return self._load_kafka(query, limit)
        elif self.type == "s3":
            return self._load_s3(query, limit)
        elif self.type == "csv_file":
            return self._load_csv()
        elif self.type == "json_file":
            return self._load_json()
        elif self.type == "parquet":
            return self._load_parquet(limit)
        return [], []

    def _load_sql(self, query, limit):
        if self.type == "clickhouse":
            q = query or f"SELECT * FROM {self.params.get('table', 'default')} LIMIT {limit}"
            result = self._conn.query(q)
            columns = result.column_names
            rows = [dict(zip(columns, row)) for row in result.result_rows[:limit]]
            return rows, columns

        cursor = self._conn.cursor()
        q = query or f"SELECT * FROM information_schema.tables LIMIT {limit}"
        cursor.execute(q)

        if self.type == "sqlite":
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = [dict(zip(columns, row)) for row in cursor.fetchmany(limit)]
        elif self.type in ("mysql",):
            columns = [d[0] for d in cursor.description] if cursor.description else []
            rows = cursor.fetchmany(limit)
        else:
            columns = [d[0] for d in cursor.description] if cursor.description else []
            raw = cursor.fetchmany(limit)
            rows = [dict(zip(columns, row)) for row in raw]
        cursor.close()
        return rows, columns

    def _sink(self, rows, target):
        cat = self.config["category"]
        if cat == "sql":
            return self._sink_sql(rows, target)
        elif self.type == "mongodb":
            return self._sink_mongo(rows, target)
        elif self.type == "elasticsearch":
            return self._sink_es(rows, target)
        elif self.type == "kafka":
            return self._sink_kafka(rows, target)
        elif self.type == "csv_file":
            return self._sink_csv(rows, target)
        elif self.type == "json_file":
            return self._sink_json(rows, target)
        return 0

    def _sink_sql(self, rows, table):
        if not rows or not table:
            return 0
        columns = list(rows[0].keys())
        placeholders = ", ".join(["%s"] * len(columns))
        col_str = ", ".join(columns)
        cursor = self._conn.cursor()
        for row in rows:
            vals = tuple(row.get(c) for c in columns)
            cursor.execute(f"INSERT INTO {table} ({col_str}) VALUES ({placeholders})", vals)
        self._conn.commit()
        cursor.close()
        return len(rows)

    def _list_tables(self):
        cat = self.config["category"]
        if cat == "sql":
            return self._list_tables_sql()
        elif self.type == "mongodb":
            return self._list_collections_mongo()
        elif self.type == "elasticsearch":
            return self._list_indices_es()
        elif self.type == "kafka":
            return self._list_topics_kafka()
        return []

    def _list_tables_sql(self):
        cursor = self._conn.cursor()
        if self.type == "sqlite":
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
        elif self.type == "clickhouse":
            result = self._conn.query("SHOW TABLES")
            tables = [row[0] for row in result.result_rows]
        else:
            cursor.execute("SHOW TABLES" if self.type == "mysql" else
                          "SELECT table_name FROM information_schema.tables WHERE table_schema='public'" if self.type == "postgresql" else
                          "SELECT table_name FROM all_tables WHERE owner=USER" if self.type == "oracle" else
                          "SELECT name FROM sys.tables")
            tables = [row[0] if isinstance(row, (list, tuple)) else list(row.values())[0] for row in cursor.fetchall()]
        cursor.close()
        return tables

    # ── MongoDB ──

    def _connect_mongo(self):
        from pymongo import MongoClient
        self._conn = MongoClient(self.params["uri"])
        self._db = self._conn[self.params["database"]]

    def _load_mongo(self, query, limit):
        coll = self.params.get("collection", "")
        if not coll:
            return [], []
        q = json.loads(query) if query else {}
        docs = list(self._db[coll].find(q).limit(limit))
        for d in docs:
            d["_id"] = str(d["_id"])
        columns = list(docs[0].keys()) if docs else []
        return docs, columns

    def _sink_mongo(self, rows, target):
        coll = target or self.params.get("collection", "output")
        result = self._db[coll].insert_many(rows)
        return len(result.inserted_ids)

    def _list_collections_mongo(self):
        return self._db.list_collection_names()

    # ── Redis ──

    def _connect_redis(self):
        import redis as r
        p = self.params
        self._conn = r.Redis(host=p["host"], port=int(p.get("port", 6379)),
            password=p.get("password") or None, db=int(p.get("db", 0)), decode_responses=True)

    def _load_redis(self, query, limit):
        pattern = query or "*"
        keys = self._conn.keys(pattern)[:limit]
        rows = []
        for k in keys:
            t = self._conn.type(k)
            val = None
            if t == "string":
                val = self._conn.get(k)
            elif t == "hash":
                val = self._conn.hgetall(k)
            elif t == "list":
                val = self._conn.lrange(k, 0, -1)
            rows.append({"key": k, "type": t, "value": val})
        return rows, ["key", "type", "value"]

    # ── Elasticsearch ──

    def _connect_es(self):
        from elasticsearch import Elasticsearch
        p = self.params
        kwargs = {"hosts": [p["hosts"]]}
        if p.get("api_key"):
            kwargs["api_key"] = p["api_key"]
        self._conn = Elasticsearch(**kwargs)

    def _load_es(self, query, limit):
        idx = self.params.get("index", "")
        q = json.loads(query) if query else {"match_all": {}}
        resp = self._conn.search(index=idx, query=q, size=limit)
        rows = [hit["_source"] for hit in resp["hits"]["hits"]]
        columns = list(rows[0].keys()) if rows else []
        return rows, columns

    def _sink_es(self, rows, target):
        idx = target or self.params.get("index", "output")
        from elasticsearch.helpers import bulk
        actions = [{"_index": idx, "_source": row} for row in rows]
        success, _ = bulk(self._conn, actions)
        return success

    def _list_indices_es(self):
        return list(self._conn.indices.get_alias(index="*").keys())

    # ── Kafka ──

    def _connect_kafka(self):
        pass  # Kafka uses producer/consumer per operation

    def _load_kafka(self, query, limit):
        from kafka import KafkaConsumer
        p = self.params
        consumer = KafkaConsumer(
            p["topic"],
            bootstrap_servers=p["bootstrap_servers"],
            group_id=p.get("group_id", "sg_consumer"),
            auto_offset_reset=p.get("auto_offset_reset", "earliest"),
            value_deserializer=lambda m: json.loads(m.decode("utf-8")),
            consumer_timeout_ms=5000,
        )
        rows = []
        for msg in consumer:
            rows.append(msg.value if isinstance(msg.value, dict) else {"value": msg.value})
            if len(rows) >= limit:
                break
        consumer.close()
        columns = list(rows[0].keys()) if rows else []
        return rows, columns

    def _sink_kafka(self, rows, target):
        from kafka import KafkaProducer
        p = self.params
        producer = KafkaProducer(
            bootstrap_servers=p["bootstrap_servers"],
            value_serializer=lambda v: json.dumps(v).encode("utf-8"),
        )
        topic = target or p["topic"]
        for row in rows:
            producer.send(topic, value=row)
        producer.flush()
        producer.close()
        return len(rows)

    def _list_topics_kafka(self):
        from kafka import KafkaConsumer
        consumer = KafkaConsumer(bootstrap_servers=self.params["bootstrap_servers"])
        topics = list(consumer.topics())
        consumer.close()
        return topics

    # ── S3 ──

    def _connect_s3(self):
        import boto3
        p = self.params
        kwargs = {}
        if p.get("endpoint_url"):
            kwargs["endpoint_url"] = p["endpoint_url"]
        if p.get("aws_access_key_id"):
            kwargs["aws_access_key_id"] = p["aws_access_key_id"]
            kwargs["aws_secret_access_key"] = p["aws_secret_access_key"]
        if p.get("region"):
            kwargs["region_name"] = p["region"]
        self._conn = boto3.client("s3", **kwargs)

    def _load_s3(self, query, limit):
        bucket = self.params["bucket"]
        prefix = self.params.get("prefix", "")
        resp = self._conn.list_objects_v2(Bucket=bucket, Prefix=prefix, MaxKeys=limit)
        rows = [{"key": obj["Key"], "size": obj["Size"], "modified": str(obj["LastModified"])}
                for obj in resp.get("Contents", [])]
        return rows, ["key", "size", "modified"]

    # ── File Sources ──

    def _load_csv(self):
        p = self.params
        with open(p["path"], "r", encoding=p.get("encoding", "utf-8")) as f:
            reader = csv_mod.DictReader(f, delimiter=p.get("delimiter", ","))
            rows = list(reader)
        columns = list(rows[0].keys()) if rows else []
        return rows, columns

    def _sink_csv(self, rows, target):
        path = target or self.params["path"]
        if not rows:
            return 0
        columns = list(rows[0].keys())
        with open(path, "w", newline="", encoding="utf-8") as f:
            writer = csv_mod.DictWriter(f, fieldnames=columns)
            writer.writeheader()
            writer.writerows(rows)
        return len(rows)

    def _load_json(self):
        p = self.params
        text = Path(p["path"]).read_text(encoding="utf-8")
        if p.get("lines"):
            rows = [json.loads(line) for line in text.strip().split("\n") if line.strip()]
        else:
            data = json.loads(text)
            rows = data if isinstance(data, list) else [data]
        columns = list(rows[0].keys()) if rows else []
        return rows, columns

    def _sink_json(self, rows, target):
        path = target or self.params["path"]
        if self.params.get("lines"):
            with open(path, "w", encoding="utf-8") as f:
                for row in rows:
                    f.write(json.dumps(row, default=str) + "\n")
        else:
            Path(path).write_text(json.dumps(rows, indent=2, default=str))
        return len(rows)

    def _load_parquet(self, limit):
        import pyarrow.parquet as pq
        table = pq.read_table(self.params["path"])
        df = table.to_pandas()
        if limit:
            df = df.head(limit)
        rows = df.to_dict(orient="records")
        columns = list(df.columns)
        return rows, columns
