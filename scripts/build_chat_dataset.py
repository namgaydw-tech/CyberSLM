# ruff: noqa: E501 -- long authored examples are kept as readable string literals.

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

USER = "<|user|>\n"
ASSISTANT = "<|assistant|>\n"
END = "\n<|end|>"

GENERAL_TERMS = [
    (
        "machine learning",
        "a way for computers to learn patterns from examples instead of following only hand-written rules",
        ["patterns", "examples"],
    ),
    (
        "artificial intelligence",
        "the field of building systems that perform tasks associated with reasoning, perception, language, or decision support",
        ["systems", "tasks"],
    ),
    (
        "neural network",
        "a layered mathematical model that adjusts weights to map inputs to useful outputs",
        ["layers", "weights"],
    ),
    (
        "training data",
        "the examples used to adjust a model's parameters during learning",
        ["examples", "parameters"],
    ),
    (
        "validation data",
        "held-out examples used to compare settings and choose checkpoints without training on them",
        ["held-out", "checkpoint"],
    ),
    (
        "test data",
        "a final unseen set used only to estimate performance after choices are fixed",
        ["unseen", "performance"],
    ),
    (
        "overfitting",
        "learning training examples too specifically, which harms performance on new inputs",
        ["training", "new"],
    ),
    (
        "underfitting",
        "failing to learn enough structure from the training data",
        ["learn", "structure"],
    ),
    (
        "inference",
        "using a trained model to produce an output for a new input",
        ["trained", "output"],
    ),
    (
        "embedding",
        "a numeric vector that represents an item so useful relationships can be learned",
        ["vector", "represents"],
    ),
    (
        "token",
        "a unit of text processed by a language model; CyberSLM uses individual UTF-8 bytes",
        ["unit", "byte"],
    ),
    (
        "context window",
        "the limited number of preceding tokens a model can consider at one time",
        ["tokens", "limited"],
    ),
    (
        "Transformer",
        "a neural architecture that uses attention to combine information across token positions",
        ["attention", "tokens"],
    ),
    (
        "attention",
        "a mechanism that weights which earlier token representations matter for the current computation",
        ["weights", "token"],
    ),
    (
        "RoPE",
        "rotary positional embeddings that rotate query and key features so attention can represent relative position",
        ["position", "query", "key"],
    ),
    (
        "RMSNorm",
        "a normalization method that rescales activations using their root-mean-square magnitude",
        ["normalize", "activations"],
    ),
    (
        "SwiGLU",
        "a gated feed-forward activation that combines a SiLU gate with a learned value projection",
        ["gate", "feed-forward"],
    ),
    (
        "gradient descent",
        "an optimization process that changes parameters in the direction that reduces loss",
        ["parameters", "loss"],
    ),
    (
        "learning rate",
        "the step size used when an optimizer updates model parameters",
        ["step", "parameters"],
    ),
    (
        "loss function",
        "a numerical measure of how far model predictions are from training targets",
        ["measure", "predictions"],
    ),
]

CYBER_TERMS = [
    (
        "cybersecurity anomaly",
        "an event or pattern that differs from an expected baseline and deserves investigation, but does not by itself prove an attack",
        ["baseline", "not", "attack"],
    ),
    (
        "phishing",
        "a deceptive message designed to trick someone into revealing information or taking an unsafe action",
        ["deceptive", "message"],
    ),
    (
        "malware",
        "software intentionally created to disrupt, damage, spy on, or gain unauthorized access to systems",
        ["software", "unauthorized"],
    ),
    (
        "ransomware",
        "malware that blocks access to data or systems and demands payment",
        ["malware", "payment"],
    ),
    (
        "firewall",
        "a control that allows or blocks network traffic according to defined rules",
        ["network", "rules"],
    ),
    (
        "intrusion detection",
        "monitoring that looks for suspicious activity and raises signals for investigation",
        ["monitoring", "investigation"],
    ),
    (
        "vulnerability",
        "a weakness that could be exploited to affect confidentiality, integrity, or availability",
        ["weakness", "exploited"],
    ),
    (
        "patch",
        "a software update intended to correct defects or security weaknesses",
        ["update", "weakness"],
    ),
    (
        "least privilege",
        "giving an account only the access needed for its current task",
        ["access", "needed"],
    ),
    (
        "multi-factor authentication",
        "requiring two or more independent forms of verification before granting access",
        ["verification", "access"],
    ),
    (
        "hash",
        "a fixed-size value computed from data, commonly used for integrity checks and secure password storage designs",
        ["integrity", "value"],
    ),
    (
        "encryption",
        "transforming readable data into protected ciphertext using a key",
        ["ciphertext", "key"],
    ),
    (
        "zero trust",
        "an approach that continuously verifies access rather than trusting a request only because of network location",
        ["verify", "access"],
    ),
    (
        "incident response",
        "the organized process of preparing for, detecting, containing, eradicating, and recovering from security incidents",
        ["contain", "recover"],
    ),
    (
        "security log",
        "a time-stamped record of system or network activity used for monitoring and investigation",
        ["record", "investigation"],
    ),
    (
        "false positive",
        "a benign event incorrectly flagged as suspicious or malicious",
        ["benign", "flagged"],
    ),
    (
        "threat model",
        "a structured description of assets, adversaries, attack paths, and mitigations",
        ["assets", "mitigations"],
    ),
    (
        "attack surface",
        "the collection of exposed points through which a system could be attacked",
        ["exposed", "points"],
    ),
    (
        "backup",
        "a separate recoverable copy of data that helps restore operations after loss or corruption",
        ["copy", "restore"],
    ),
    (
        "network segmentation",
        "dividing a network into controlled zones to limit unnecessary access and lateral movement",
        ["zones", "access"],
    ),
]

PROGRAMMING_TERMS = [
    (
        "variable",
        "a named reference to a value that a program can read or update",
        ["named", "value"],
    ),
    (
        "function",
        "a reusable block of code that accepts inputs and may return an output",
        ["reusable", "output"],
    ),
    (
        "loop",
        "a control structure that repeats an operation while a condition or sequence requires it",
        ["repeats", "condition"],
    ),
    (
        "conditional",
        "logic that chooses which code to run based on a true-or-false expression",
        ["chooses", "true"],
    ),
    ("list", "an ordered collection that stores multiple values", ["ordered", "values"]),
    ("dictionary", "a collection that maps unique keys to values", ["keys", "values"]),
    (
        "class",
        "a definition that groups data and behavior for creating related objects",
        ["data", "objects"],
    ),
    (
        "object",
        "an instance that contains state and can provide behavior defined by its type",
        ["instance", "state"],
    ),
    (
        "exception",
        "a signal that an unusual or error condition interrupted normal execution",
        ["error", "execution"],
    ),
    (
        "unit test",
        "a small automated check of one behavior in isolation",
        ["automated", "behavior"],
    ),
    (
        "API",
        "a defined interface that lets software components exchange requests and responses",
        ["interface", "requests"],
    ),
    (
        "JSON",
        "a text format for representing objects, arrays, strings, numbers, booleans, and null",
        ["text", "objects"],
    ),
    (
        "algorithm",
        "a finite sequence of steps for solving a problem or computing a result",
        ["steps", "problem"],
    ),
    (
        "data structure",
        "an organized way to store and access data for particular operations",
        ["store", "access"],
    ),
    (
        "recursion",
        "a technique where a function solves a problem by calling itself on smaller cases",
        ["function", "smaller"],
    ),
    (
        "version control",
        "a system that records file changes so work can be reviewed, compared, and restored",
        ["changes", "restored"],
    ),
    (
        "Git branch",
        "an independent line of version-control work that can later be reviewed and merged",
        ["independent", "merged"],
    ),
    (
        "dependency",
        "external code or a package that a project needs to operate",
        ["package", "project"],
    ),
    (
        "debugging",
        "the process of reproducing, locating, and correcting software defects",
        ["locating", "defects"],
    ),
    (
        "complexity",
        "a description of how an algorithm's time or memory needs grow with input size",
        ["time", "input"],
    ),
]

CAPABILITIES = [
    (
        "explain a cybersecurity term",
        "I can give a short educational explanation and state important uncertainty or safety limits",
        ["explain", "cybersecurity"],
    ),
    (
        "write a small Python example",
        "I can suggest a compact example and explain its main parts, but it should still be tested",
        ["Python", "tested"],
    ),
    (
        "review structured event counts",
        "I can describe supplied counts; deterministic calculations should be used when exact arithmetic matters",
        ["counts", "deterministic"],
    ),
    (
        "confirm a real attack",
        "I cannot confirm an attack from a few aggregates alone; source evidence and human investigation are required",
        ["cannot", "investigation"],
    ),
    (
        "suggest beginner projects",
        "I can propose scoped project ideas, learning goals, and safe next steps",
        ["project", "safe"],
    ),
    (
        "replace a security analyst",
        "I cannot replace an analyst; I am an educational prototype that can assist with explanations",
        ["cannot", "analyst"],
    ),
    (
        "remember an unlimited conversation",
        "I cannot remember unlimited history because my verified context is 512 byte tokens",
        ["512", "context"],
    ),
    (
        "browse live threat intelligence",
        "I do not have live browsing in the local model and may be outdated",
        ["not", "live"],
    ),
    (
        "help debug code",
        "I can reason about small code snippets and suggest checks, but generated fixes require testing",
        ["code", "testing"],
    ),
    (
        "explain machine learning",
        "I can explain basic machine-learning ideas in short language",
        ["machine", "explain"],
    ),
    (
        "calculate exact percentages",
        "The neural model may make arithmetic mistakes; use the labeled deterministic mode for verified calculations",
        ["deterministic", "mistakes"],
    ),
    (
        "identify malware from one log line",
        "I cannot reliably identify malware from one line; more context and validated tools are needed",
        ["cannot", "context"],
    ),
    (
        "provide legal advice",
        "I am not a legal adviser and can only offer general educational information",
        ["not", "legal"],
    ),
    (
        "provide medical advice",
        "I am not a medical professional and should not be used for diagnosis",
        ["not", "medical"],
    ),
    (
        "summarize a short discussion",
        "I can summarize the portion of a conversation that still fits in my verified context window",
        ["summarize", "context"],
    ),
    (
        "generate passwords",
        "I should recommend a password manager or secure generator rather than inventing reusable passwords in chat",
        ["password", "manager"],
    ),
    (
        "teach safe networking basics",
        "I can explain defensive networking concepts without pretending that text alone verifies a system",
        ["network", "defensive"],
    ),
    (
        "answer every factual question",
        "I cannot guarantee every fact; I should acknowledge uncertainty when evidence is missing",
        ["cannot", "uncertainty"],
    ),
    (
        "use private project data",
        "Private data should stay local unless its license and release authorization are recorded",
        ["private", "authorization"],
    ),
    (
        "describe my architecture",
        "I can describe my 35-million-parameter Transformer, byte tokenizer, RoPE, RMSNorm, and SwiGLU",
        ["Transformer", "RoPE"],
    ),
]

IDEA_AREAS = [
    "password hygiene",
    "phishing awareness",
    "log analysis",
    "network inventory",
    "file integrity",
    "secure coding",
    "backup verification",
    "access control",
    "web security headers",
    "malware indicators",
    "privacy settings",
    "patch tracking",
    "authentication",
    "threat modeling",
    "incident timelines",
    "DNS monitoring",
    "USB device policy",
    "email safety",
    "home router security",
    "security education",
]

INSUFFICIENT = [
    (
        "Is this IP address malicious?",
        "the address, time range, source, and corroborating evidence",
    ),
    (
        "Did this login prove an account takeover?",
        "login location, device history, authentication evidence, and user confirmation",
    ),
    (
        "Is this process malware?",
        "the process name, path, hash, parent process, host context, and trusted analysis",
    ),
    (
        "Did the firewall stop the attack?",
        "the relevant rule, traffic logs, timeline, and affected systems",
    ),
    ("Is this email phishing?", "the sender, headers, links, attachments, and message context"),
    ("Should I delete this file?", "the file identity, source, purpose, hash, and backup status"),
    (
        "Was data exfiltrated?",
        "network flows, destination ownership, transferred volume, endpoint logs, and timeline",
    ),
    (
        "Is the server secure?",
        "its configuration, exposed services, patch level, access controls, and threat model",
    ),
    (
        "Who caused this alert?",
        "validated attribution evidence; one alert cannot identify an actor",
    ),
    (
        "Can I ignore this warning?",
        "the warning text, affected component, severity source, and operational context",
    ),
    (
        "Is the vulnerability exploitable?",
        "the exact version, configuration, exposure, prerequisites, and vendor guidance",
    ),
    (
        "Did encryption fail?",
        "the algorithm, mode, key handling, error output, and implementation details",
    ),
    (
        "Is my password strong?",
        "its generation method and length; do not paste the actual password",
    ),
    (
        "Was this download safe?",
        "the source, signature, hash, file type, and sandbox or scanner results",
    ),
    (
        "Did the backup work?",
        "restore-test results, timestamps, scope, integrity checks, and storage status",
    ),
    (
        "Is this port dangerous?",
        "the service, exposure, authentication, configuration, and business need",
    ),
    (
        "Can this anomaly confirm ransomware?",
        "endpoint evidence, file changes, process activity, notes, and corroborating alerts",
    ),
    (
        "Is the code vulnerability fixed?",
        "the patch diff, tests, affected paths, and security review evidence",
    ),
    (
        "Should this user be blocked?",
        "identity evidence, policy, risk, business context, and an authorized reviewer",
    ),
    (
        "Is the certificate trustworthy?",
        "the hostname, chain, validity period, issuer, revocation status, and intended use",
    ),
]

QUESTION_WRAPPERS = [
    "What does {term} mean?",
    "Explain {term} simply.",
    "Give me a short explanation of {term}.",
    "I am a beginner. What is {term}?",
    "Can you define {term} without jargon?",
    "Help me understand {term}.",
    "In one short paragraph, explain {term}.",
    "What should a student know about {term}?",
]
ANSWER_WRAPPERS = [
    "{definition}.",
    "In simple terms, {definition}.",
    "A useful way to think about it is this: {definition}.",
    "Briefly, {definition}.",
    "For a beginner: {definition}.",
    "The core idea is that it is {definition}.",
    "It means {definition}.",
    "A concise explanation is: {definition}.",
]


def normalized(text: str) -> str:
    return re.sub(r"\s+", " ", text.strip().lower())


def render_prompt(user: str, history: list[tuple[str, str]] | None = None) -> str:
    parts: list[str] = []
    for old_user, old_assistant in history or []:
        parts.extend((USER, old_user, "\n", ASSISTANT, old_assistant, END, "\n"))
    parts.extend((USER, user, "\n", ASSISTANT))
    return "".join(parts)


def scenarios() -> dict[str, list[dict[str, Any]]]:
    result: dict[str, list[dict[str, Any]]] = {}
    result["general_explanation"] = [
        {"id": f"general_{i:02d}", "term": term, "definition": definition, "terms": terms}
        for i, (term, definition, terms) in enumerate(GENERAL_TERMS)
    ]
    result["cybersecurity_explanation"] = [
        {"id": f"cyber_{i:02d}", "term": term, "definition": definition, "terms": terms}
        for i, (term, definition, terms) in enumerate(CYBER_TERMS)
    ]
    result["programming_explanation"] = [
        {"id": f"programming_{i:02d}", "term": term, "definition": definition, "terms": terms}
        for i, (term, definition, terms) in enumerate(PROGRAMMING_TERMS)
    ]
    greetings = []
    salutations = ["Hi", "Hello", "Good morning", "Good evening", "Hey CyberSLM"]
    contexts = [
        "I am starting a project",
        "I am learning cybersecurity",
        "I have a quick question",
        "How are you",
    ]
    for index, (salutation, context) in enumerate(
        pair for salutation in salutations for pair in [(salutation, value) for value in contexts]
    ):
        greetings.append(
            {
                "id": f"greeting_{index:02d}",
                "user": f"{salutation}. {context}?",
                "answer": (
                    f"Hello! I am CyberSLM, a small educational assistant. I can help with a "
                    f"short question about AI, programming, or defensive cybersecurity related to {context.lower()}."
                ),
                "terms": ["CyberSLM", "help"],
            }
        )
    result["greeting"] = greetings
    result["capability"] = [
        {"id": f"capability_{i:02d}", "task": task, "answer": answer, "terms": terms}
        for i, (task, answer, terms) in enumerate(CAPABILITIES)
    ]
    result["idea_generation"] = [
        {
            "id": f"idea_{i:02d}",
            "area": area,
            "answer": (
                f"Three beginner projects for {area} are: 1) build a fictional checklist, "
                f"2) analyze a small synthetic dataset, and 3) create a local dashboard that explains findings. "
                "Use only authorized systems and document what each result does not prove."
            ),
            "terms": ["1)", "2)", "3)"],
        }
        for i, area in enumerate(IDEA_AREAS)
    ]
    result["insufficient_information"] = [
        {
            "id": f"insufficient_{i:02d}",
            "question": question,
            "answer": f"I do not have enough information to conclude that. Please provide {missing}.",
            "terms": ["not", "enough", "information"],
        }
        for i, (question, missing) in enumerate(INSUFFICIENT)
    ]
    structured = []
    event_types = ["EVENT_CONNECT", "EVENT_READ", "EVENT_WRITE", "EVENT_OPEN", "EVENT_SENDTO"]
    for index in range(20):
        first = event_types[index % len(event_types)]
        second = event_types[(index + 2) % len(event_types)]
        first_count = 3 + index
        second_count = 1 + (index * 3) % 17
        winner = first if first_count > second_count else second
        structured.append(
            {
                "id": f"structured_{index:02d}",
                "user": (
                    f"Evidence counts: {first}={first_count}, {second}={second_count}. "
                    "Which event type is most frequent, and does that alone confirm an attack?"
                ),
                "answer": (
                    f"{winner} is most frequent. These counts alone do not confirm an attack; "
                    "they are observations that require source evidence and investigation."
                ),
                "terms": [winner, "not", "confirm"],
            }
        )
    result["structured_evidence"] = structured
    multiturn = []
    for index, area in enumerate(IDEA_AREAS):
        suggestion = f"create a small local {area} checklist using fictional examples"
        explanation = (
            f"Start by defining three goals for {area}, add five fictional test cases, and record "
            "expected versus observed results. Keep it local and finish with limitations."
        )
        multiturn.append(
            {
                "id": f"multiturn_{index:02d}",
                "history": [
                    (f"Suggest one beginner project about {area}.", f"You could {suggestion}.")
                ],
                "user": "Explain that suggestion in a little more detail.",
                "answer": explanation,
                "terms": [area.split()[0], "fictional", "limitations"],
            }
        )
    result["multi_turn"] = multiturn
    return result


def render_record(
    category: str, scenario: dict[str, Any], variant: int
) -> tuple[str, str, list[str]]:
    if category in {"general_explanation", "cybersecurity_explanation", "programming_explanation"}:
        user = QUESTION_WRAPPERS[variant % len(QUESTION_WRAPPERS)].format(term=scenario["term"])
        answer = [
            "{term} means {definition}.",
            "In simple terms, {term} is {definition}.",
            "A short definition: {term} is {definition}.",
            "Think of {term} as {definition}.",
        ][variant % 4].format(term=scenario["term"], definition=scenario["definition"])
    elif category == "greeting":
        user = [
            "{user}",
            "{user} Please reply briefly.",
            "{user} It is nice to meet you.",
            "{user} Can we start with a short greeting?",
            "{user} I am beginning a study session.",
            "{user} Please introduce yourself in one sentence.",
            "{user} I am ready to learn.",
            "{user} Let us begin our conversation.",
        ][variant % 8].format(user=scenario["user"])
        answer = [
            "Hello! I am CyberSLM. {answer}",
            "Hi! {answer}",
            "Welcome. {answer}",
            "Nice to meet you. {answer}",
        ][variant % 4].format(answer=scenario["answer"])
    elif category == "capability":
        user = [
            "Can you {task}?",
            "Are you able to {task}?",
            "Tell me whether you can {task}.",
            "What are your limits when asked to {task}?",
            "Could you safely {task}?",
            "Describe how you would {task}.",
            "Is {task} within your capabilities?",
            "How could you assist a student who wants to {task}?",
        ][variant % 8].format(task=scenario["task"])
        answer = [
            "For {task}, {answer}.",
            "My scope is clear here: {answer}.",
            "I can answer cautiously: {answer}.",
            "The important limit is this: {answer}.",
        ][variant % 4].format(task=scenario["task"], answer=scenario["answer"])
    elif category == "idea_generation":
        user = [
            "Suggest three beginner projects about {area}.",
            "Give me three safe project ideas for {area}.",
            "What can a beginner build to learn {area}?",
            "List three small educational {area} projects.",
            "Propose three fictional practice activities about {area}.",
            "Name three ethical learning exercises for {area}.",
            "How could I explore {area} through three small projects?",
            "Give a numbered list of three introductory {area} ideas.",
        ][variant % 8].format(area=scenario["area"])
        answer = [
            "{answer}",
            "Here are scoped ideas. {answer}",
            "A safe beginner list is: {answer}",
            "For {area}, try this plan. {answer}",
        ][variant % 4].format(area=scenario["area"], answer=scenario["answer"])
    elif category == "insufficient_information":
        user = [
            "{question}",
            "Please answer carefully: {question}",
            "Based only on that fact, {question}",
            "Without inventing evidence, {question}",
            "State any uncertainty when answering: {question}",
            "Can the available information settle this question: {question}",
            "Give a cautious assessment: {question}",
            "Explain what more evidence is needed. {question}",
        ][variant % 8].format(question=scenario["question"])
        answer = [
            "{answer}",
            "A cautious answer is: {answer}",
            "That conclusion is not supported. {answer}",
            "More evidence is required. {answer}",
        ][variant % 4].format(answer=scenario["answer"])
    elif category == "structured_evidence":
        user = [
            "{question}",
            "Use the stated fictional evidence only. {question}",
            "Calculate this exactly: {question}",
            "For this synthetic example, {question}",
            "Give the result without adding observations. {question}",
            "Check the arithmetic in this question: {question}",
            "Answer from the supplied counts. {question}",
            "Provide a concise evidence-based answer. {question}",
        ][variant % 8].format(question=scenario["user"])
        answer = [
            "{answer}",
            "From the supplied fictional counts, {answer}",
            "The evidence-based result is: {answer}",
            "Using only those counts, {answer}",
        ][variant % 4].format(answer=scenario["answer"])
    elif category == "multi_turn":
        follow_up = [
            scenario["user"],
            "Could you expand that first suggestion?",
            "What steps would I follow for that idea?",
            "Explain how a beginner could start it.",
            "What is a safe first version of that project?",
            "Please add practical steps to your suggestion.",
            "How should I scope that idea for a student?",
            "Explain that project with a small workflow.",
        ][variant % 8]
        answer = [
            "{answer}",
            "For that project, {answer}",
            "A practical workflow is: {answer}",
            "Keep the scope small: {answer}",
        ][variant % 4].format(answer=scenario["answer"])
        return render_prompt(follow_up, scenario["history"]), answer + END, scenario["terms"]
    else:
        raise ValueError(category)
    return render_prompt(user), answer + END, scenario["terms"]


def acceptance_records() -> list[dict[str, Any]]:
    prompts = [
        ("Hi CyberSLM.", ["hi", "hello"]),
        ("What can you help me with?", ["cybersecurity", "programming"]),
        ("Explain machine learning in simple terms.", ["patterns", "examples"]),
        ("What is a cybersecurity anomaly?", ["baseline", "attack"]),
        ("Suggest three beginner cybersecurity projects.", ["1", "2", "3"]),
        ("Explain your first suggestion.", ["project", "start"]),
        ("What does RoPE do in a Transformer?", ["position", "attention"]),
        ("Can 15 network events confirm an attack?", ["not", "confirm"]),
        ("How are you different from a general LLM?", ["small", "CyberSLM"]),
        ("Can you summarize what we discussed?", ["discussed", "summary"]),
    ]
    return [
        {
            "qa_id": f"acceptance_{index:02d}",
            "conversation_id": "acceptance_conversation",
            "split": "acceptance",
            "category": "acceptance",
            "prompt": render_prompt(question),
            "answer": "",
            "rubric_terms_json": json.dumps(terms),
            "synthetic": True,
            "source_id": "project_authored_acceptance_prompt",
        }
        for index, (question, terms) in enumerate(prompts, start=1)
    ]


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description="Build the fictional CyberSLM chat corpus")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=2041)
    args = parser.parse_args()
    rng = random.Random(args.seed)
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    all_records: dict[str, list[dict[str, Any]]] = {"train": [], "validation": [], "test": []}
    split_variants = {"train": 4, "validation": 4, "test": 4}
    scenario_groups: dict[str, list[str]] = {}
    for category, raw_scenarios in scenarios().items():
        ordered = raw_scenarios.copy()
        rng.shuffle(ordered)
        offsets = {"train": (0, 14), "validation": (14, 17), "test": (17, 20)}
        for split, (start, end) in offsets.items():
            selected = ordered[start:end]
            scenario_groups[f"{category}:{split}"] = [item["id"] for item in selected]
            for scenario in selected:
                for variant in range(split_variants[split]):
                    prompt, answer, terms = render_record(category, scenario, variant)
                    all_records[split].append(
                        {
                            "qa_id": f"{scenario['id']}_{variant:02d}",
                            "conversation_id": scenario["id"],
                            "split": split,
                            "category": category,
                            "prompt": prompt,
                            "answer": answer,
                            "rubric_terms_json": json.dumps(terms),
                            "synthetic": True,
                            "source_id": "project_authored_synthetic_chat_v1",
                        }
                    )
    prompt_seen: set[str] = set()
    answer_seen: set[str] = set()
    split_hashes: dict[str, str] = {}
    for split, records in all_records.items():
        rng.shuffle(records)
        for record in records:
            prompt_key = normalized(record["prompt"])
            pair_key = normalized(record["prompt"] + "\n" + record["answer"])
            if prompt_key in prompt_seen or pair_key in answer_seen:
                raise ValueError(f"Duplicate conversation detected: {record['qa_id']}")
            prompt_seen.add(prompt_key)
            answer_seen.add(pair_key)
        path = output / f"{split}_chat.parquet"
        pd.DataFrame(records).to_parquet(path, index=False)
        split_hashes[split] = sha256_file(path)
    acceptance = output / "acceptance_chat.parquet"
    pd.DataFrame(acceptance_records()).to_parquet(acceptance, index=False)
    split_hashes["acceptance"] = sha256_file(acceptance)
    case_sets = {
        split: {record["conversation_id"] for record in records}
        for split, records in all_records.items()
    }
    if any(
        case_sets[left] & case_sets[right]
        for left, right in (("train", "validation"), ("train", "test"), ("validation", "test"))
    ):
        raise ValueError("Conversation groups overlap across splits")
    manifest = {
        "dataset_id": "cyberslm_chat_synthetic_v2",
        "status": "PASS",
        "seed": args.seed,
        "license": "project-authored synthetic fixture; release authorized by project owner",
        "provenance": "No third-party conversation corpus and no private event records.",
        "role_format": {"user": USER.strip(), "assistant": ASSISTANT.strip(), "end": END.strip()},
        "tokenizer": "existing CyberSLM UTF-8 byte tokenizer; no tokenizer training",
        "split_examples": {split: len(records) for split, records in all_records.items()}
        | {"acceptance": 10},
        "split_conversations": {split: len(groups) for split, groups in case_sets.items()},
        "split_sha256": split_hashes,
        "category_counts": {
            split: dict(Counter(record["category"] for record in records))
            for split, records in all_records.items()
        },
        "scenario_groups": scenario_groups,
        "checks": {
            "conversation_group_overlap": 0,
            "duplicate_prompts": 0,
            "duplicate_prompt_answer_pairs": 0,
            "all_synthetic": True,
            "max_prompt_answer_utf8_bytes": max(
                len((record["prompt"] + record["answer"]).encode("utf-8"))
                for records in all_records.values()
                for record in records
            ),
        },
    }
    (output / "dataset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    (output / "validation_report.json").write_text(
        json.dumps({"status": "PASS", "checks": manifest["checks"]}, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(manifest, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
