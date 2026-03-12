# services/openai_client.py

import os
from pathlib import Path
from typing import Optional

from dotenv import load_dotenv
from openai import OpenAI

load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class OpenAIClient:
    """
    Thin wrapper around the OpenAI Python SDK.

    Purpose:
    - keep API setup in one place
    - make orchestrator code cleaner
    - standardize model calls across your app
    """

    def __init__(
        self,
        model: Optional[str] = None,
        api_key: Optional[str] = None,
    ) -> None:
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.model = model or os.getenv("OPENAI_MODEL", "gpt-4o")

        if not self.api_key:
            raise ValueError(
                "OPENAI_API_KEY is missing. Add it to your .env file or environment variables."
            )

        self.client = OpenAI(api_key=self.api_key)

    def generate_text(
        self,
        user_input: str,
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
    ) -> str:
        """
        Basic text generation for your copilot.

        Args:
            user_input: the prompt or business question
            system_prompt: optional behavior instruction
            temperature: optional sampling control

        Returns:
            Model output as plain text
        """
        kwargs = {
            "model": self.model,
            "input": user_input,
        }

        if system_prompt:
            kwargs["instructions"] = system_prompt

        if temperature is not None:
            kwargs["temperature"] = temperature

        response = self.client.responses.create(**kwargs)
        return response.output_text.strip()

    def summarize_supply_chain_issue(
        self,
        question: str,
        evidence: str,
    ) -> str:
        """
        Tailored helper for root-cause and action-oriented answers.
        Good fit for your orchestrator.
        """
        system_prompt = """
You are a supply chain planning copilot.

Your job:
- analyze the business question
- use the provided evidence only
- explain likely root causes clearly
- recommend practical next actions
- keep the answer concise and business-friendly

Return in this structure:

1. Executive Summary
2. Likely Root Causes
3. Recommended Actions
4. Risks / Watchouts
"""

        user_input = f"""
Business question:
{question}

Evidence:
{evidence}
"""

        return self.generate_text(
            user_input=user_input,
            system_prompt=system_prompt,
            temperature=0.2,
        )

    def draft_email(
        self,
        recipient_role: str,
        subject_context: str,
        body_context: str,
    ) -> str:
        """
        Drafts a business email based on supply chain findings.
        """
        system_prompt = f"""
You are an operations assistant drafting a professional email to a {recipient_role}.

Rules:
- be concise
- be professional
- be action-oriented
- include a clear subject line
- avoid fluff
"""

        user_input = f"""
Context for the email subject:
{subject_context}

Context for the email body:
{body_context}

Draft the full email.
"""

        return self.generate_text(
            user_input=user_input,
            system_prompt=system_prompt,
            temperature=0.3,
        )


# Optional singleton-style helper
_openai_client_instance: Optional[OpenAIClient] = None


def get_openai_client() -> OpenAIClient:
    """
    Reuse one client instance across the app.
    """
    global _openai_client_instance

    if _openai_client_instance is None:
        _openai_client_instance = OpenAIClient()

    return _openai_client_instance