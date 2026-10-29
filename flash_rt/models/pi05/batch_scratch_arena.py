"""Fixed-address scratch buffers shared by serialized batch graphs."""

class BatchScratchArena:
    def __init__(self, max_batch_size, allocator, zero_allocator=None):
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self._allocator = allocator
        self._zero_allocator = zero_allocator
        self._buffers = {}

    def acquire(self, name, count, dtype, *, batch_size, extra_rows=0, zeroed=False):
        batch_size, count, extra_rows = int(batch_size), int(count), int(extra_rows)
        rows = batch_size + extra_rows
        if batch_size < 1 or batch_size > self.max_batch_size or extra_rows < 0 or count <= 0 or count % rows:
            raise ValueError("scratch size must scale exactly with a supported batch")
        per_sample = count // rows
        key = str(name)
        if key in self._buffers:
            buf, prior_count, prior_dtype, prior_extra, prior_zeroed = self._buffers[key]
            if prior_count != per_sample or prior_dtype != dtype or prior_extra != extra_rows or prior_zeroed != zeroed:
                raise ValueError("scratch layout changed after graph capture")
            return buf
        allocator = self._zero_allocator if zeroed else self._allocator
        if allocator is None:
            raise ValueError("zero allocator is required")
        buf = allocator(per_sample * (self.max_batch_size + extra_rows), dtype)
        self._buffers[key] = (buf, per_sample, dtype, extra_rows, zeroed)
        return buf

    def acquire_fixed(self, name, count, dtype, *, zeroed=False):
        key = "fixed:" + str(name)
        layout = (int(count), dtype, bool(zeroed))
        if key in self._buffers:
            buf, prior = self._buffers[key]
            if prior != layout:
                raise ValueError("fixed scratch layout changed after graph capture")
            return buf
        allocator = self._zero_allocator if zeroed else self._allocator
        if allocator is None or count <= 0:
            raise ValueError("invalid fixed scratch allocator or size")
        buf = allocator(count, dtype)
        self._buffers[key] = (buf, layout)
        return buf
