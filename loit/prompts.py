from .data import stable_text


PROPOSAL_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "string", "minLength": 1} for name in ["hypothesis", "competing_explanation", "discriminating_test", "prediction", "competing_prediction"]},
    "required": ["hypothesis", "competing_explanation", "discriminating_test", "prediction", "competing_prediction"],
    "additionalProperties": False,
}
GATE_SCHEMA = {
    "type": "object",
    "properties": {name: {"type": "number", "minimum": 0, "maximum": 1} for name in ["relevance", "consistency", "novelty", "test_quality"]},
    "required": ["relevance", "consistency", "novelty", "test_quality"],
    "additionalProperties": False,
}


def focused(state):
    return (f"You are a scientific research agent.\nQuestion: {state.question}.\n"
            f"Observations up to this point: {stable_text(state.observations)}.\n"
            f"Proposals up to this point: {stable_text(state.proposals)}.\n"
            f"Tools: {stable_text(state.tools)}.\nTake the next step. Answer with one action.")


def activity(instruction):
    return f"Activity: {instruction}.\nWrite one short response to the activity."


def proposal():
    return ("Before the next action, propose a scientific hypothesis and a discriminating test. "
            "Return a JSON object with hypothesis, competing_explanation, discriminating_test, "
            "prediction, and competing_prediction. Each prediction states the observation expected "
            "under its corresponding explanation.")


def gate(state, candidate):
    return (f"Question: {state.question}.\nObservations: {stable_text(state.observations)}.\n"
            f"Prior proposals: {stable_text(state.proposals)}.\nProposal: {candidate.text()}.\n"
            "Rate relevance to the question, agreement with the observations, novelty against prior "
            "proposals, and the discriminative quality of the test from 0 to 1. Return a JSON object "
            "with relevance, consistency, novelty, and test_quality.")
