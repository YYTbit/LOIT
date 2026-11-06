import importlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import time


def resolve(specification):
    module, name = specification.split(":", 1)
    return getattr(importlib.import_module(module), name)


def load_environments(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(manifest, dict) or not manifest:
        raise ValueError("Environment configuration requires named factories")
    result = {}
    for name, entry in manifest.items():
        if not isinstance(name, str) or not name:
            raise ValueError("Environment names require nonempty strings")
        environment = resolve(entry["factory"])(**entry.get("kwargs", {}))
        if not callable(getattr(environment, "start", None)):
            raise ValueError("An environment factory requires a start(task, seed) method")
        result[name] = environment
    return result


class CommandSession:
    def __init__(self, command, task, seed, timeout):
        self.timeout = timeout
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=None, bufsize=0, env=environment)
        os.set_blocking(self.process.stdin.fileno(), False)
        os.set_blocking(self.process.stdout.fileno(), False)
        self._pending = bytearray()
        self._failed = False
        try:
            reply = self.request("reset", task=task, seed=seed)
            self.initial_observation, self.tools = reply["observation"], reply.get("tools", [])
        except Exception:
            self.close()
            raise

    def request(self, method, **values):
        if self._failed:
            raise RuntimeError("The scientific benchmark interface is unavailable")
        try:
            deadline = time.monotonic() + self.timeout
            outgoing = memoryview((json.dumps({"method": method, **values}, allow_nan=False) + "\n").encode("utf-8"))
            with selectors.DefaultSelector() as selector:
                selector.register(self.process.stdin, selectors.EVENT_WRITE)
                selector.register(self.process.stdout, selectors.EVENT_READ)
                # Readiness covers available bytes, so retain partial lines under the same deadline.
                while outgoing or b"\n" not in self._pending:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        raise TimeoutError("The scientific benchmark interface timed out")
                    events = selector.select(remaining)
                    if not events:
                        raise TimeoutError("The scientific benchmark interface timed out")
                    for key, _ in events:
                        try:
                            if key.fileobj is self.process.stdin:
                                written = os.write(key.fd, outgoing[:65536])
                                outgoing = outgoing[written:]
                                if not outgoing:
                                    selector.unregister(self.process.stdin)
                            else:
                                content = os.read(key.fd, 65536)
                                if not content:
                                    raise RuntimeError("The scientific benchmark interface terminated before a complete reply")
                                self._pending.extend(content)
                        except BlockingIOError:
                            continue
            line, _, remaining = self._pending.partition(b"\n")
            self._pending = bytearray(remaining)
            try:
                reply = json.loads(line)
            except (json.JSONDecodeError, UnicodeDecodeError) as error:
                raise RuntimeError("The scientific benchmark interface returned invalid JSON") from error
            if not isinstance(reply, dict):
                raise RuntimeError("The scientific benchmark interface requires a JSON object")
        except BaseException:
            self._failed = True
            raise
        if "error" in reply:
            raise RuntimeError(reply["error"])
        return reply

    def step(self, action):
        result = self.request("step", action=action)
        self.tools = result.get("tools", self.tools)
        return result["observation"], result["done"]

    def score(self):
        return self.request("score")["score"]

    def close(self):
        self._failed = True
        if self.process.poll() is None:
            # EOF lets an external worker release its native resources.
            self.process.stdin.close()
            try:
                self.process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                self.process.terminate()
                try:
                    self.process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    self.process.kill()
                    self.process.wait()
        self.process.stdin.close()
        self.process.stdout.close()


class CommandEnvironment:
    def __init__(self, command, timeout=120):
        if not isinstance(command, list) or not command or any(not isinstance(value, str) for value in command):
            raise ValueError("An environment command requires an argument list")
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("An environment timeout requires a positive finite value")
        self.command, self.timeout = command, timeout

    def start(self, task, seed):
        return CommandSession(self.command, task, seed, self.timeout)


class IdeationSession:
    def __init__(self, task, scorer):
        self.task, self.scorer = task, scorer
        self.initial_observation = task.get("context", "")
        self.answer = None

    def step(self, action):
        self.answer = action
        return "Scientific proposal recorded.", True

    def score(self):
        if self.answer is None:
            return 0.0
        if self.scorer is None:
            raise ValueError("Native scoring requires the released benchmark evaluator")
        return float(self.scorer(self.task, self.answer))

    def close(self):
        pass


class IdeationEnvironment:
    def __init__(self, scorer):
        self.scorer = resolve(scorer)

    def start(self, task, seed):
        return IdeationSession(task, self.scorer)


def evaluate_collection(benchmark, predictions, evaluator):
    if not isinstance(benchmark, str) or not benchmark:
        raise ValueError("Evaluation requires a benchmark identifier")
    if not predictions:
        raise ValueError("Scientific evaluation requires prediction records")
    # Collection-level metrics, including native fluency, are delegated as a complete collection.
    scores = resolve(evaluator)(predictions)
    if not isinstance(scores, dict) or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores.values()):
        raise ValueError("The released evaluator requires finite named metrics")
    return scores
