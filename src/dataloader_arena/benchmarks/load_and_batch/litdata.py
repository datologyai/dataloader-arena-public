"""Load-and-batch benchmark entrypoint for LitData."""

import json
import sys
from typing import Any, Iterable, Iterator

from dataloader_arena.context import LoaderConfig
from dataloader_arena.loaders import litdata as base


def create_loader(config: LoaderConfig) -> Iterable[Any] | Iterator[Any]:
    """Instantiate the LitData loader for benchmarks (includes DataLoader)."""

    from litdata import StreamingDataLoader

    loader_config, dataset, dataset_opts, dataloader_opts = base.build_dataset(config)

    def _jsonify(o: object) -> str:
        return getattr(o, "__qualname__", getattr(o, "__name__", repr(o)))

    if config.verbose:
        print(
            f"[INFO] DataLoader opts for LitData\n{json.dumps(dict(dataset_opts), indent=4, default=_jsonify)}",
            file=sys.stderr,
        )

    dataloader = StreamingDataLoader(
        dataset,
        **dataloader_opts,
    )

    if config.verbose:
        print(
            f"[INFO] StreamingDataLoader opts for LitData\n{json.dumps(dict(dataloader_opts), indent=4, default=_jsonify)}",
            file=sys.stderr,
        )

    serializable_dataset_opts = {
        k: (
            repr(v)
            if not isinstance(v, (str, int, float, bool, list, dict, type(None)))
            else v
        )
        for k, v in dataset_opts.items()
    }
    dataloader.__arena_resolved_config__ = {  # type: ignore[attr-defined]
        "parsed": loader_config.model_dump(),
        "dataset_options": serializable_dataset_opts,
        "dataloader_options": dataloader_opts,
        "run": {
            "batch_size": config.batch_size,
            "num_workers": config.num_workers,
            "seed": config.seed,
            "limit_steps": config.limit_steps,
            "verbose": config.verbose,
        },
        "paths": {
            "local": str(config.local_path),
            "remote": config.remote_path,
        },
    }
    return dataloader
