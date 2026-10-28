"""Fixed-address scratch buffers shared by serialized batch graphs."""

class BatchScratchArena:
    def __init__(self, max_batch_size, allocator):
        self.max_batch_size = int(max_batch_size)
        if self.max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        self._allocator = allocator
        self._buffers = {}

    def acquire(self, name, count, dtype, *, batch_size):
        batch_size, count = int(batch_size), int(count)
        if batch_size < 1 or batch_size > self.max_batch_size or count <= 0 or count % batch_size:
            raise ValueError("scratch size must scale exactly with a supported batch")
        per_sample = count // batch_size
        key = str(name)
        if key in self._buffers:
            buf, prior_count, prior_dtype = self._buffers[key]
            if prior_count != per_sample or prior_dtype != dtype:
                raise ValueError("scratch layout changed after graph capture")
            return buf
        buf = self._allocator(per_sample * self.max_batch_size, dtype)
        self._buffers[key] = (buf, per_sample, dtype)
        return buf
