"""
Llama API Client Wrapper

Handles structured output with Pydantic models.
Adapted from Kaggle_Human_In_The_Loop project.
"""

import json
import os
import time
from typing import Any, Type, TypeVar

from dotenv import load_dotenv
from pydantic import BaseModel

load_dotenv()

T = TypeVar("T", bound=BaseModel)


def dereference_schema(
    schema: dict[str, Any], defs: dict[str, Any] | None = None
) -> dict[str, Any]:
    """Recursively dereference a JSON schema by inlining all $ref references."""
    if defs is None:
        defs = schema.get("$defs", {})

    if not isinstance(schema, dict):
        return schema

    if "$ref" in schema:
        ref_path = schema["$ref"]
        if ref_path.startswith("#/$defs/"):
            ref_name = ref_path.split("/")[-1]
            if ref_name in defs:
                return dereference_schema(defs[ref_name].copy(), defs)
        return schema

    result = {}

    for key, value in schema.items():
        if key == "$defs":
            continue

        if key == "anyOf":
            dereffed_options = []
            for opt in value:
                dereffed = dereference_schema(opt, defs)
                dereffed_options.append(dereffed)
            result["anyOf"] = dereffed_options

        elif isinstance(value, dict):
            result[key] = dereference_schema(value, defs)
        elif isinstance(value, list):
            result[key] = [
                dereference_schema(item, defs) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            result[key] = value

    return result


def clean_schema_for_llama(schema: dict[str, Any]) -> dict[str, Any]:
    """Clean a Pydantic JSON schema for Llama API compatibility."""
    dereffed = dereference_schema(schema)
    return _clean_properties(dereffed)


def _clean_properties(schema: dict[str, Any]) -> dict[str, Any]:
    """Remove unsupported properties recursively."""
    if not isinstance(schema, dict):
        return schema

    cleaned = {}

    for key, value in schema.items():
        if key == "default":
            continue

        if isinstance(value, dict):
            if "anyOf" in value:
                cleaned_value = {}
                for k, v in value.items():
                    if k in ("anyOf", "title", "description"):
                        if k == "anyOf":
                            cleaned_value[k] = [_clean_properties(opt) for opt in v]
                        else:
                            cleaned_value[k] = v
                cleaned[key] = cleaned_value
            else:
                cleaned[key] = _clean_properties(value)
        elif isinstance(value, list):
            cleaned[key] = [
                _clean_properties(item) if isinstance(item, dict) else item
                for item in value
            ]
        else:
            cleaned[key] = value

    return cleaned


class LlamaClient:
    """
    Wrapper for Llama API with structured output support.
    """

    MODEL = "Llama-4-Maverick-17B-128E-Instruct-FP8"
    BASE_URL = "https://api.llama.com/v1"

    def __init__(self, api_key: str | None = None):
        self.api_key = api_key or os.getenv("LLAMA_API_KEY")
        if not self.api_key:
            raise ValueError(
                "LLAMA_API_KEY not found. Set it in .env file or pass to constructor."
            )

        try:
            import httpx

            self._httpx = httpx
        except ImportError:
            raise ImportError("httpx is required. Install with: pip install httpx")

    def _get_clean_schema(self, schema: Type[T]) -> dict[str, Any]:
        """Get a Llama-compatible JSON schema from a Pydantic model."""
        raw_schema = schema.model_json_schema()
        return clean_schema_for_llama(raw_schema)

    def generate_structured(
        self,
        schema: Type[T],
        system_prompt: str,
        user_prompt: str,
        temperature: float = 0.1,
        max_retries: int = 3,
    ) -> T:
        """
        Generate a structured response using a Pydantic schema.

        Args:
            schema: Pydantic model class defining the expected output structure
            system_prompt: System message for the LLM
            user_prompt: User message for the LLM
            temperature: Sampling temperature (default 0.1 for consistency)
            max_retries: Number of retries on transient errors

        Returns:
            Validated Pydantic model instance
        """
        headers = {
            "Authorization": f"Bearer {self.api_key}",
            "Content-Type": "application/json",
        }

        clean_json_schema = self._get_clean_schema(schema)

        payload = {
            "model": self.MODEL,
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
            "temperature": temperature,
            "max_completion_tokens": 16000,
            "response_format": {
                "type": "json_schema",
                "json_schema": {
                    "name": schema.__name__,
                    "schema": clean_json_schema,
                },
            },
        }

        last_error = None
        for attempt in range(max_retries):
            try:
                with self._httpx.Client(timeout=180.0) as client:
                    response = client.post(
                        f"{self.BASE_URL}/chat/completions",
                        headers=headers,
                        json=payload,
                    )
                    response.raise_for_status()

                result = response.json()
                content = result["completion_message"]["content"]["text"]

                try:
                    data = json.loads(content)
                except json.JSONDecodeError as e:
                    print(f"  JSON parse error: {e}")
                    print(
                        f"  Response was truncated or malformed. Retrying... ({attempt + 1}/{max_retries})"
                    )
                    last_error = e
                    if attempt < max_retries - 1:
                        time.sleep(2)
                        continue
                    print(f"  Raw response (first 500 chars): {content[:500]}...")
                    raise ValueError(
                        f"LLM returned invalid JSON after {max_retries} attempts: {e}"
                    )

                return schema.model_validate(data)

            except (
                self._httpx.ReadError,
                self._httpx.ConnectError,
                self._httpx.TimeoutException,
            ) as e:
                last_error = e
                if attempt < max_retries - 1:
                    wait_time = 2**attempt
                    print(
                        f"  Connection error, retrying in {wait_time}s... ({attempt + 1}/{max_retries})"
                    )
                    time.sleep(wait_time)
                    continue
                raise

        raise last_error
