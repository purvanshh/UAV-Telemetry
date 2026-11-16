# From the ARJUNA dataset tools (tools/arjuna_converter/ulog_writer.py, by Labhesh), unchanged except for this
# header. Used to write CSV telemetry as ARJUNA ULog, and by the tests to build synthetic flights.
"""Minimal ULog v1 writer that follows the PX4 ULog file format and the
ARJUNA Flight Logging Standard v1.1 integrity rules (arjuna_log_hash chain).

Record order: header, B (flag bits), I (info), F (formats), P (params),
then A/D/L/S data records in time order.
"""
import hashlib
import struct

ULOG_MAGIC = b"ULog\x01\x12\x35"
SYNC_MAGIC = bytes([0x2F, 0x73, 0x13, 0x20, 0x25, 0x0C, 0xBB, 0x12])

# ULog type -> (struct code, size)
TYPES = {
    "int8_t": ("b", 1), "uint8_t": ("B", 1), "int16_t": ("h", 2),
    "uint16_t": ("H", 2), "int32_t": ("i", 4), "uint32_t": ("I", 4),
    "int64_t": ("q", 8), "uint64_t": ("Q", 8), "float": ("f", 4),
    "double": ("d", 8), "bool": ("?", 1), "char": ("c", 1),
}

HASH_BLOCK = 65536

LOG_TOPIC = ("arjuna_log_hash", [
    ("uint64_t", "timestamp", 1),
    ("uint32_t", "block_index", 1),
    ("uint64_t", "block_end_offset", 1),
    ("uint8_t", "chain_hash", 32),
])


class Topic:
    def __init__(self, name, fields):
        # fields: list of (type, name, array_len)
        self.name = name
        self.fields = fields
        if fields[0][1] != "timestamp" or fields[0][0] != "uint64_t":
            raise ValueError(f"{name}: first field must be uint64_t timestamp")
        code = "<"
        for t, _, n in fields:
            c = TYPES[t][0]
            code += (str(n) + c) if n > 1 else c
        self.struct = struct.Struct(code)

    def format_string(self):
        parts = []
        for t, n, cnt in self.fields:
            parts.append(f"{t}[{cnt}] {n}" if cnt > 1 else f"{t} {n}")
        return self.name + ":" + ";".join(parts) + ";"

    def pack(self, values):
        """values: dict field_name -> scalar or sequence."""
        flat = []
        for t, n, cnt in self.fields:
            v = values[n]
            if cnt > 1:
                v = list(v)
                if len(v) != cnt:
                    raise ValueError(f"{self.name}.{n}: expected {cnt} values, got {len(v)}")
                flat.extend(v)
            else:
                flat.append(v)
        return self.struct.pack(*flat)


def _rec(rtype, payload):
    return struct.pack("<HB", len(payload), ord(rtype)) + payload


def _kv(type_name, key, value_bytes):
    k = f"{type_name} {key}".encode()
    return struct.pack("<B", len(k)) + k + value_bytes


class ULogWriter:
    def __init__(self, path, start_timestamp):
        self.f = open(path, "wb")
        self.offset = 0
        self.topics = {}
        self.subs = {}       # (topic, multi_id) -> msg_id
        self.next_msg_id = 0
        self.chain = b"\x00" * 32
        self.block_start = 0
        self.block_buf = bytearray()
        self.block_index = 0
        self.last_ts = start_timestamp
        self.last_sync_ts = start_timestamp
        self.hash_topic = Topic(*LOG_TOPIC)
        self._write(ULOG_MAGIC + bytes([1]) + struct.pack("<Q", start_timestamp))
        # B record: compat flags[8], incompat flags[8], appended offsets[3]
        self._write(_rec("B", bytes(8) + bytes(8) + bytes(24)))
        self._phase = "definitions"

    # ---- low level -------------------------------------------------------
    def _write(self, data):
        self.f.write(data)
        self.offset += len(data)
        self.block_buf += data
        if self._phase_data() and len(self.block_buf) >= HASH_BLOCK:
            self._close_block()

    def _phase_data(self):
        return getattr(self, "_phase", "") == "data"

    def _close_block(self):
        block = bytes(self.block_buf)
        self.block_buf = bytearray()
        self.chain = hashlib.sha256(self.chain + block).digest()
        payload = self.hash_topic.pack({
            "timestamp": self.last_ts, "block_index": self.block_index,
            "block_end_offset": self.offset, "chain_hash": list(self.chain),
        })
        self.block_index += 1
        msg_id = self.subs[("arjuna_log_hash", 0)]
        # this record belongs to the next block
        rec = _rec("D", struct.pack("<H", msg_id) + payload)
        self.f.write(rec)
        self.offset += len(rec)
        self.block_buf += rec

    # ---- definitions -----------------------------------------------------
    def info(self, key, value):
        if isinstance(value, str):
            b = value.encode()
            self._write(_rec("I", _kv(f"char[{len(b)}]", key, b)))
        elif isinstance(value, tuple):  # (ulog_type, value)
            t, v = value
            self._write(_rec("I", _kv(t, key, struct.pack("<" + TYPES[t][0], v))))
        else:
            raise TypeError(key)

    def add_topic(self, topic):
        self.topics[topic.name] = topic

    def write_formats(self):
        self.add_topic(self.hash_topic)
        for t in self.topics.values():
            self._write(_rec("F", t.format_string().encode()))

    def param(self, name, value):
        if isinstance(value, float):
            self._write(_rec("P", _kv("float", name, struct.pack("<f", value))))
        else:
            self._write(_rec("P", _kv("int32_t", name, struct.pack("<i", int(value)))))

    def begin_data(self):
        self._phase = "data"
        self.subscribe("arjuna_log_hash", 0)

    # ---- data ------------------------------------------------------------
    def subscribe(self, topic, multi_id):
        key = (topic, multi_id)
        if key in self.subs:
            return self.subs[key]
        mid = self.next_msg_id
        self.next_msg_id += 1
        self.subs[key] = mid
        self._write(_rec("A", struct.pack("<BH", multi_id, mid) + topic.encode()))
        return mid

    def data(self, topic, multi_id, values):
        ts = values["timestamp"]
        self._maybe_sync(ts)
        mid = self.subscribe(topic, multi_id)
        payload = self.topics[topic].pack(values)
        self.last_ts = max(self.last_ts, ts)
        self._write(_rec("D", struct.pack("<H", mid) + payload))

    def log(self, level, timestamp, text):
        """level: 0 emerg .. 7 debug (stored as ASCII digit per ULog)."""
        self._maybe_sync(timestamp)
        b = text.encode("utf-8", "replace")
        self._write(_rec("L", struct.pack("<BQ", ord(str(level)), timestamp) + b))

    def _maybe_sync(self, ts):
        if ts - self.last_sync_ts >= 1_000_000:
            self.last_sync_ts = ts
            self._write(_rec("S", SYNC_MAGIC))

    def close(self):
        # Close the final (partial) block so the whole file is covered,
        # except the very last hash record itself (as on the drone).
        if self.block_buf:
            self._close_block()
        self.f.close()
        return {
            "file_size_bytes": self.offset,
            "block_count": self.block_index,
            "final_chain_hash": self.chain.hex(),
        }
