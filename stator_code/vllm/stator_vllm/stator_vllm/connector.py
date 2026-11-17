import ctypes
import json
import logging
import mmap
import os

from vllm.distributed.kv_transfer.kv_connector.v1.base import KVConnectorBase_V1, KVConnectorMetadata
from vllm.v1.core.kv_cache_utils import BlockHash, make_block_hash_with_group_id

import stator_vllm

logger = logging.getLogger("stator_vllm")


class StatorMetadata(KVConnectorMetadata):
    pass


class StatorConnector(KVConnectorBase_V1):
    @property
    def prefer_cross_layer_blocks(self) -> bool:
        return True

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config=vllm_config, role=role, kv_cache_config=kv_cache_config)
        self.pool = None
        self.published = False

    def start_load_kv(self, forward_context, **kwargs) -> None:
        return

    def wait_for_layer_load(self, layer_name: str) -> None:
        return

    def save_kv_layer(self, layer_name, kv_layer, attn_metadata, **kwargs) -> None:
        return

    def wait_for_save(self):
        return

    def get_num_new_matched_tokens(self, request, num_computed_tokens):
        return 0, False

    def update_state_after_alloc(self, request, blocks, num_external_tokens):
        return

    def build_connector_meta(self, scheduler_output) -> KVConnectorMetadata:
        return StatorMetadata()

    def bind_gpu_block_pool(self, gpu_block_pool) -> None:
        self.pool = gpu_block_pool
        state = stator_vllm.kv_state
        if state.get("role") != "agent":
            return
        for position, digest in enumerate(state["hashes"]):
            block = gpu_block_pool.blocks[1 + position]
            gpu_block_pool.touch([block])
            key = make_block_hash_with_group_id(BlockHash(bytes.fromhex(digest)), 0)
            block.block_hash = key
            gpu_block_pool.cached_block_hash_to_block.insert(key, block)
        logger.warning("stator: kv_attach blocks=%d tokens=%d", len(state["hashes"]), len(state["hashes"]) * 16)

    def request_finished(self, request, block_ids):
        state = stator_vllm.kv_state
        if state.get("role") != "publisher" or self.published or self.pool is None:
            return False, None
        hashes = list(request.block_hashes)
        count = min(len(hashes), len(block_ids))
        if count == 0 or list(block_ids[:count]) != list(range(1, count + 1)):
            logger.warning("stator: kv_publish blocks=0 reason=first_request_not_in_first_blocks")
            return False, None
        self.published = True
        self.pool.touch([self.pool.blocks[block_id] for block_id in block_ids[:count]])
        block_bytes = state["block_bytes"]
        stator_vllm._libc.mprotect(ctypes.c_void_p(state["base"] + block_bytes), count * block_bytes,
                                   mmap.PROT_READ)
        description = {"file": state["file"], "blocks": count, "block_bytes": block_bytes,
                       "bytes": state["bytes"], "prompt_tokens": request.num_prompt_tokens,
                       "hashes": [bytes(digest).hex() for digest in hashes[:count]]}
        with open(state["file"] + ".json.writing", "w") as index:
            json.dump(description, index)
        os.replace(state["file"] + ".json.writing", state["file"] + ".json")
        state["blocks"] = count
        logger.warning("stator: kv_publish blocks=%d tokens=%d file=%s", count, count * 16, state["file"])
        return False, None
