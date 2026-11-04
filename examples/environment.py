"""Minimal session example for checking the environment interface."""


class ExampleSession:
    def __init__(self, task):
        self.initial_observation = task.get("observations", [])
        self.tools = []
        self.answer = None

    def step(self, action):
        self.answer = action
        return "Proposal recorded.", True

    def score(self):
        return 0.0

    def close(self):
        pass


class ExampleEnvironment:
    data_kind = "software_fixture"

    def start(self, task, seed):
        return ExampleSession(task)
