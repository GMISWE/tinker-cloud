"""The server's log line for a finished pass request (forward, forward_backward, optim_step): what
the request carried and where its seconds went. One line per request, written by the task manager."""
import re
from dataclasses import dataclass
from typing import Optional

PREFIX = "request"
_FIELD = re.compile(r"(\w+)=(\S+)")


@dataclass(frozen=True)
class RequestSize:
    """What a pass request carried."""
    datums: Optional[int]   # None: an optim_step, which carries no datum
    body_bytes: int         # the HTTP body as it arrived


@dataclass(frozen=True)
class RequestLine:
    request_id: str
    operation: str
    model_id: str
    outcome: str            # "completed" or "failed"
    size: RequestSize
    queued_s: float         # from the request's row in the store to the start of its work
    backend_s: float        # in the service and the backend
    store_s: float          # the rest: writing the request's row, then checking and storing its result

    def format(self) -> str:
        datums = "none" if self.size.datums is None else str(self.size.datums)
        return (f"{PREFIX} id={self.request_id} op={self.operation} model={self.model_id} outcome={self.outcome} "
                f"datums={datums} body_bytes={self.size.body_bytes} "
                f"queued_s={self.queued_s:.3f} backend_s={self.backend_s:.3f} store_s={self.store_s:.3f}")

    @classmethod
    def parse(cls, log_line: str) -> Optional["RequestLine"]:
        """The request line inside `log_line` (a logger may put its own text before it); None when
        `log_line` holds none."""
        start = log_line.find(f"{PREFIX} id=")
        if start < 0:
            return None
        fields = dict(_FIELD.findall(log_line[start:]))
        return cls(
            request_id=fields["id"], operation=fields["op"], model_id=fields["model"], outcome=fields["outcome"],
            size=RequestSize(datums=None if fields["datums"] == "none" else int(fields["datums"]),
                             body_bytes=int(fields["body_bytes"])),
            queued_s=float(fields["queued_s"]), backend_s=float(fields["backend_s"]), store_s=float(fields["store_s"]),
        )
