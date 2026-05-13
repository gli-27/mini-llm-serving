"""Locust load test suite for the Mini LLM Serving Platform.

Three user profiles with different traffic patterns:
- NormalUser (70%): Standard completions with mixed priorities
- StreamUser (20%): SSE streaming completions
- BurstUser (10%): Burst traffic — 10 rapid requests per cycle

Usage:
    locust -f tests/load/locustfile.py --host http://localhost:8000

Interview: "I test three traffic patterns because real production traffic
is a mix: most requests are normal, some are streaming (chat UX), and
occasionally a batch job fires a burst. Rate limiting and load shedding
should handle all three gracefully — 429/503 responses are expected and
counted as successes since they indicate the system is protecting itself."
"""

import random

from locust import HttpUser, between, tag, task

PROMPTS = [
    "Hello, how are you?",
    "Explain quantum computing",
    "Write a haiku about servers",
    "What is backpropagation?",
    "Tell me about distributed systems",
    "How does TCP work?",
    "Explain gradient descent",
    "What is a hash table?",
]

# Status codes that indicate the system is functioning correctly:
# 200 = success, 429 = rate limited, 503 = load shedding / circuit breaker
ACCEPTABLE_CODES = {200, 429, 503}


class NormalUser(HttpUser):
    """Standard user — sends one completion at a time with mixed priorities.

    Weight 7: represents 70% of traffic.
    """

    weight = 7
    wait_time = between(1, 3)

    @task
    @tag("completions", "normal")
    def create_completion(self) -> None:
        """POST /v1/completions with random prompt and priority."""
        with self.client.post(
            "/v1/completions",
            json={
                "prompt": random.choice(PROMPTS),
                "max_tokens": 50,
                "priority": random.randint(1, 3),
            },
            catch_response=True,
        ) as resp:
            if resp.status_code in ACCEPTABLE_CODES:
                resp.success()


class StreamUser(HttpUser):
    """Streaming user — requests SSE token-by-token responses.

    Weight 2: represents 20% of traffic.
    """

    weight = 2
    wait_time = between(2, 5)

    @task
    @tag("completions", "streaming")
    def create_streaming_completion(self) -> None:
        """POST /v1/completions with stream=True."""
        with self.client.post(
            "/v1/completions",
            json={
                "prompt": random.choice(PROMPTS),
                "max_tokens": 100,
                "stream": True,
                "priority": 2,
            },
            catch_response=True,
            stream=True,
        ) as resp:
            if resp.status_code in ACCEPTABLE_CODES:
                # Consume stream to measure full response time
                if resp.status_code == 200:
                    text = resp.text
                    if "data:" in text:
                        resp.success()
                    else:
                        resp.failure("No SSE data in streaming response")
                else:
                    resp.success()


class BurstUser(HttpUser):
    """Burst traffic — fires 10 rapid requests per cycle.

    Weight 1: represents 10% of traffic.
    Simulates batch job or automated pipeline hitting the API.
    """

    weight = 1
    wait_time = between(5, 10)

    @task
    @tag("completions", "burst")
    def burst_requests(self) -> None:
        """Fire 10 rapid requests without waiting between them."""
        for _ in range(10):
            with self.client.post(
                "/v1/completions",
                json={
                    "prompt": random.choice(PROMPTS),
                    "max_tokens": 30,
                    "priority": 3,  # Batch priority
                },
                catch_response=True,
            ) as resp:
                if resp.status_code in ACCEPTABLE_CODES:
                    resp.success()
