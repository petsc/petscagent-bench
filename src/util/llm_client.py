import os
import json
from typing import Any, Dict, Optional, Type
from pydantic import BaseModel
import litellm
from litellm import acompletion
import dotenv

USE_ASKSAGE = False # do not change unless you know how to use ANL AskSage

dotenv.load_dotenv()


class LLMClient:
    """Client for making LLM API calls."""

    def __init__(self, model: str = "gpt-4o-mini", temperature: float = 0.3, api_base_url: Optional[str] = None, max_tokens: Optional[int] = None):
        """Initialize LLM client.

        Args:
            model: Model name (supports OpenAI, Anthropic, etc. via LiteLLM)
            temperature: Sampling temperature (0-1)
            api_base_url: Optional API base URL (e.g., "https://api.openai.com/v1").
                         If None, uses LiteLLM default based on model provider.
            max_tokens: Optional cap on completion length. Pin this when the
                         endpoint's default is low enough to truncate replies.
        """
        self.model = model
        self.temperature = temperature
        self.api_base_url = api_base_url
        self.max_tokens = max_tokens

    async def structured_completion(
        self,
        prompt: str,
        response_model: Type[BaseModel],
        system_prompt: Optional[str] = None,
    ) -> BaseModel:
        """Get structured response from LLM.

        Args:
            prompt: User prompt
            response_model: Pydantic model for response structure
            system_prompt: Optional system prompt

        Returns:
            Parsed response as response_model instance
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        try:
            # Use JSON mode
            completion_kwargs = {
                'model': self.model,
                'messages': messages,
                'temperature': self.temperature,
                'response_format': {"type": "json_object"},
                'timeout': 300,
            }
            if self.max_tokens:
                completion_kwargs['max_tokens'] = int(self.max_tokens)
            litellm.ssl_verify = False
            if self.api_base_url:
                completion_kwargs['api_base'] = self.api_base_url
                is_asksage_endpoint = self.api_base_url.startswith('https://api.asksage.anl.gov')
                is_argo_endpoint = 'inside.anl.gov/argoapi' in self.api_base_url
                if is_asksage_endpoint:
                    litellm.ssl_verify = os.environ["ASKSAGE_SSL_CERT_FILE"]
                    completion_kwargs['api_key'] = os.environ["ASKSAGE_API_KEY"]
                elif is_argo_endpoint:
                    # Argo authenticates with the caller's ANL domain username
                    completion_kwargs['api_key'] = os.environ["ARGO_API_KEY"]
            response = await acompletion(**completion_kwargs)
            # Parse JSON response
            content = response.choices[0].message.content
            if isinstance(content, str):
                # Remove markdown code block wrapper that some LLMs such as Claude add
                # This ensures we get clean JSON for parsing
                if content.startswith("```"):
                    content = content.split("```", 2)[1]
                    content = content.lstrip("json").strip()
                data = json.loads(content)
            else:
                raise TypeError(f"Expected string content from response, got {type(content)}")
            # Handle case where LLM returns array instead of object
            if isinstance(data, list) and data:
                # Use first element if it's an array of objects
                data = data[0]

            # Validate with pydantic model
            return response_model(**data)
        except json.JSONDecodeError as e:
            print(f"ERROR: Failed to parse JSON from LLM response")
            print(f"Raw content: {content[:500]}")
            raise
        except Exception as e:
            print(f"ERROR in structured_completion: {type(e).__name__}: {str(e)}")
            if 'content' in locals():
                print(f"Raw LLM response: {content[:500]}")
            if 'data' in locals():
                print(f"Parsed data type: {type(data)}, value: {str(data)[:200]}")
            raise

    async def completion(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        """Get text completion from LLM.

        Args:
            prompt: User prompt
            system_prompt: Optional system prompt

        Returns:
            Response text
        """
        messages = []

        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})

        messages.append({"role": "user", "content": prompt})

        response = await acompletion(
            model=self.model,
            messages=messages,
            temperature=self.temperature,
        )

        return response.choices[0].message.content
