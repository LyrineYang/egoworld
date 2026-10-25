"""Host-side reservations held through result commit, including writer retries."""


class MemoryBudget:
    def __init__(self, config):
        self.limit = config.max_in_flight_bytes
        self.default = config.estimated_result_bytes
        self.estimates = dict(config.stage_memory_bytes)
        self.reservations = {}
        self.peak = 0

    @staticmethod
    def key(job):
        return job["clip"]["clip_id"], job["stage"]

    @property
    def used(self):
        return sum(self.reservations.values())

    def fits(self, job):
        size = self.estimates.get(job["stage"], self.default)
        if self.limit and size > self.limit:
            if self.reservations:
                return False  # commit already-computed outputs before reporting an undersized budget
            raise ValueError(f"{job['stage']} reservation {size} exceeds max_in_flight_bytes={self.limit}; increase the budget or provide a measured stage_memory_bytes estimate")
        return not self.limit or self.used + size <= self.limit

    def reserve(self, job):
        self.reservations[self.key(job)] = self.estimates.get(job["stage"], self.default)
        self.peak = max(self.peak, self.used)

    def completed(self, job):
        if "_result_bytes" in job:
            size = job["_result_bytes"]
            self.reservations[self.key(job)] = size
            # Estimates can increase within a run, but never shrink from one easy clip.
            self.estimates[job["stage"]] = max(self.estimates.get(job["stage"], self.default), int(size * 1.25))
            self.peak = max(self.peak, self.used)

    def release(self, job):
        self.reservations.pop(self.key(job), None)


def ready_priority(ctx, stage, now, max_wait, model_priority):
    since = ctx["ready_since"][stage]
    urgent = now - since >= max_wait
    remaining = sum(state not in ("Done", "Failed", "Blocked") for state in ctx["states"].values())
    return (not urgent, since if urgent else 0., model_priority(stage), remaining, since)
