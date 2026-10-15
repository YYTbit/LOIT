import argparse
import contextlib
import importlib
import json
import math
import os
from pathlib import Path
import selectors
import subprocess
import sys


BENCHMARKS = ["discoveryworld", "scienceworld", "tomato_star", "liveideabench", "agentideabench"]


def resolve(specification):
    module, name = specification.split(":", 1)
    return getattr(importlib.import_module(module), name)


def load_environments(path):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    result = {}
    for name, entry in manifest.items():
        if name not in BENCHMARKS:
            raise ValueError("Unknown scientific benchmark")
        result[name] = resolve(entry["factory"])(**entry.get("kwargs", {}))
    return result


class CommandSession:
    def __init__(self, command, task, seed, timeout):
        self.timeout = timeout
        environment = os.environ.copy()
        package_root = str(Path(__file__).resolve().parent.parent)
        environment["PYTHONPATH"] = package_root + os.pathsep + environment.get("PYTHONPATH", "")
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=None, text=True, bufsize=1, env=environment)
        try:
            reply = self.request("reset", task=task, seed=seed)
            self.initial_observation, self.tools = reply["observation"], reply.get("tools", [])
        except Exception:
            self.close()
            raise

    def request(self, method, **values):
        self.process.stdin.write(json.dumps({"method": method, **values}) + "\n")
        self.process.stdin.flush()
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(self.timeout):
                raise TimeoutError("The scientific benchmark interface timed out")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError("The scientific benchmark interface terminated")
        reply = json.loads(line)
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
        if self.process.poll() is None:
            # EOF lets the worker close native resources, including the ScienceWorld Java gateway.
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


class NativeEnvironment:
    def __init__(self, benchmark, sdk_python=None, timeout=120, **options):
        if benchmark not in ["discoveryworld", "scienceworld"]:
            raise ValueError("The native worker supports interactive scientific environments")
        self.command = [sdk_python or sys.executable, "-m", "loit.benchmarks", "--benchmark", benchmark,
                        "--options", json.dumps(options)]
        self.timeout = timeout

    def start(self, task, seed):
        return CommandSession(self.command, task, seed, self.timeout)


class ScienceWorldSession:
    def __init__(self, task, seed, options):
        from scienceworld import ScienceWorldEnv
        self.environment = ScienceWorldEnv(serverPath=options.get("jar_path"), envStepLimit=options.get("step_limit", 100))
        parameters = task["parameters"]
        self.environment.load(parameters["task_name"], int(parameters["variation"]), parameters.get("simplification", ""))
        self.initial_observation, information = self.environment.reset()
        self.current_score = float(information.get("score", 0))
        self.tools = self.environment.getPossibleActions()

    def step(self, action):
        observation, reward, done, information = self.environment.step(action)
        self.current_score = float(information["score"])
        self.tools = self.environment.getPossibleActions()
        return observation, bool(done)

    def score(self):
        return self.current_score

    def close(self):
        self.environment.close()


class DiscoveryWorldSession:
    def __init__(self, task, seed, options):
        from discoveryworld.DiscoveryWorldAPI import DiscoveryWorldAPI
        self.environment = DiscoveryWorldAPI(threadID=int(options.get("thread_id", seed)))
        parameters = task["parameters"]
        loaded = self.environment.loadScenario(parameters["scenario"], parameters["difficulty"],
                                               int(parameters.get("world_seed", seed)), numUserAgents=1)
        if not loaded:
            raise ValueError("DiscoveryWorld could not load the requested released scenario")
        self.initial_observation = self.observation()
        self.tools = {"actions": self.environment.listKnownActions(),
                      "instructions": self.environment.additionalActionDescriptionString()}

    def observation(self):
        value = self.environment.getAgentObservation(0)
        # The SDK scorecard contains oracle knowledge and remains outside agent-visible observations.
        return {"ui": value.get("ui", {}), "errors": value.get("errors", [])}

    def step(self, action):
        try:
            parsed = json.loads(action)
        except (ValueError, TypeError):
            return {"error": "Return one JSON action using the listed action names and arguments."}, False
        if not isinstance(parsed, dict):
            return {"error": "The action requires a JSON object."}, False
        status = self.environment.performAgentAction(0, parsed)
        self.environment.tick()
        observation = self.observation()
        observation["action_status"] = status
        return observation, bool(self.environment.areTasksComplete())

    def score(self):
        scorecard = self.environment.getTaskScorecard()
        if not scorecard:
            raise ValueError("The official DiscoveryWorld scorecard is empty")
        return sum(float(task["scoreNormalized"]) for task in scorecard) / len(scorecard)

    def close(self):
        window = getattr(self.environment, "window", None)
        if window is not None:
            import pygame
            pygame.display.quit()


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
    if benchmark not in BENCHMARKS:
        raise ValueError("Unknown benchmark")
    if not predictions:
        raise ValueError("Scientific evaluation requires prediction records")
    # Collection-level metrics, including native fluency, are delegated as a complete collection.
    scores = resolve(evaluator)(predictions)
    if not isinstance(scores, dict) or any(not isinstance(value, (int, float)) or not math.isfinite(value) for value in scores.values()):
        raise ValueError("The released evaluator requires finite named metrics")
    return scores


def worker(benchmark, options):
    session = None
    for line in sys.stdin:
        try:
            request = json.loads(line)
            with contextlib.redirect_stdout(sys.stderr):
                method = request["method"]
                if method == "reset":
                    if session is not None:
                        session.close()
                    factory = ScienceWorldSession if benchmark == "scienceworld" else DiscoveryWorldSession
                    session = factory(request["task"], request["seed"], options)
                    reply = {"observation": session.initial_observation, "tools": session.tools}
                elif method == "step":
                    observation, done = session.step(request["action"])
                    reply = {"observation": observation, "done": done, "tools": session.tools}
                elif method == "score":
                    reply = {"score": session.score()}
                else:
                    raise ValueError("Unknown benchmark operation")
            print(json.dumps(reply, allow_nan=False), flush=True)
        except Exception as error:
            print(json.dumps({"error": str(error)}), flush=True)
    if session is not None:
        session.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark", choices=["discoveryworld", "scienceworld"], required=True)
    parser.add_argument("--options", default="{}")
    arguments = parser.parse_args()
    worker(arguments.benchmark, json.loads(arguments.options))
