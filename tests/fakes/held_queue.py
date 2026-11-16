"""Queue operations behind a held barrier, so a scheduler's worker sees them all at once."""
import asyncio


async def queue_behind_barrier(submit_barrier, submissions):
    """Hold the queue with `submit_barrier(fn)`, then start each submission in order. Returns the
    coroutine function that releases the barrier, and one task per submission."""
    gate = asyncio.Event()
    hold = asyncio.ensure_future(submit_barrier(gate.wait))
    tasks = []
    for s in submissions:
        tasks.append(asyncio.ensure_future(s))
        await asyncio.sleep(0)

    async def release():
        gate.set()
        await asyncio.wait_for(hold, 5)   # bounded: a worker that stops would resolve nothing
    return release, tasks


async def run_behind_barrier(submit_barrier, submissions):
    """Queue the submissions, release them, and return each one's result or exception."""
    release, tasks = await queue_behind_barrier(submit_barrier, submissions)
    await release()
    return await asyncio.wait_for(asyncio.gather(*tasks, return_exceptions=True), 5)
