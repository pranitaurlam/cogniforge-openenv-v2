---
title: Support Agent OpenEnv
emoji: 🎧
colorFrom: blue
colorTo: indigo
sdk: docker
pinned: false
tags:
  - openenv
---

# Support Agent OpenEnv

A real-world OpenEnv environment that simulates a customer support agent workflow. The agent must categorize, prioritize, and draft responses for incoming support tickets.

## Environment Description

Agents process a queue of customer support tickets, each requiring:
- **Category classification** (Account, Billing, Technical, Sales)
- **Priority assessment** (Low, Medium, High, Urgent)
- **Response drafting** — a professional reply to the customer

## Observation Space

| Field | Type | Description |
|---|---|---|
| `ticket_id` | string | Unique ticket identifier |
| `content` | string | Full ticket content (subject + body) |
| `current_status` | string | PENDING, IN_PROGRESS, or RESOLVED |
| `metadata` | dict | Optional ticket metadata |

## Action Space

| Field | Type | Description |
|---|---|---|
| `category` | string | One of: Account, Billing, Technical, Sales |
| `priority` | string | One of: Low, Medium, High, Urgent |
| `draft_response` | string | Professional reply to the customer |
| `is_done` | bool | Mark ticket as complete |

## Reward

Score is strictly between 0 and 1 (exclusive). Partial credit is given across sub-dimensions (category, priority, draft quality, keyword coverage).

## Tasks

| Task | Difficulty | Objective |
|---|---|---|
| `task_1_easy` | Easy | Classify a login issue correctly |
| `task_2_medium` | Medium | Triage a performance issue with correct category and priority |
| `task_3_hard` | Hard | Resolve a billing dispute with an accurate, keyword-rich draft |

## Setup

```bash
pip install -e .
server  # starts FastAPI server on port 7860
```

## Inference

```bash
export OPENAI_API_KEY=your_key
python inference.py
```

## Docker

```bash
docker build -t support-agent-env .
docker run -p 7860:7860 -e OPENAI_API_KEY=your_key support-agent-env
```
