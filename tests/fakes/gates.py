"""Stand-ins for the gate suite: a driver that only records deletes, and a
trace mix-in whose execute() returns canned rows (no SDK, no torch)."""


class StubDriver:
    base_url = "stub://"

    def __init__(self):
        self.deleted = []

    def delete_model(self, model_id):
        self.deleted.append(model_id)
        return True


class CannedRows:
    """Mix in before the trace class: `class StubTrace(CannedRows, SegSweep)`."""

    def __init__(self, rows, **kw):
        super().__init__(**kw)
        self._rows = rows

    def execute(self, driver):
        return [dict(r) for r in self._rows]
