from abc import ABC, abstractmethod
from app.dataset import DataItem
from app.llm import LLM
from app.llm_extractor import LLMExtractor
from app.logger import logger
from typing import Dict, List, Tuple, Optional, Callable
from app.services import get_schema_service
from app.tracing import trace_event
import re

class BaseSQLGenerator(ABC):

    def __init__(self, extractor_max_retry: Optional[int] = None):
        self._extractor_max_retry = extractor_max_retry
        self._extractor = LLMExtractor() if extractor_max_retry is None else LLMExtractor(max_retry=extractor_max_retry)

    @abstractmethod
    def generate(self, data_item: DataItem, llm: LLM, sampling_budget: int = 1) -> Tuple[List[str], Dict[str, int]]:
        pass

    def _get_extractor(self) -> LLMExtractor:
        return self._extractor
        
    def _generate_with_progressive_stripping(
        self,
        data_item: DataItem,
        llm: LLM,
        prompt_format_func: Callable[[str], str],
        sampling_budget: int = 1
    ) -> Tuple[str, int]:
        """
        Helper method to generate prompt with progressive schema stripping.
        Returns (final_prompt, compressed_level).
        """
        max_prompt_len = llm.llm_config.max_model_len - llm.llm_config.max_tokens
        schema_service = get_schema_service()
        schema_to_use = getattr(data_item, "database_schema_after_schema_linking", data_item.database_schema)

        # 逐级记录 stripping 尝试（每次格式化即一个 level，顺序与 PROFILE_STRIPPING_LEVELS 一致）
        attempts = []

        def _recording_prompt_format_func(schema_profile: str) -> str:
            formatted_prompt = prompt_format_func(schema_profile)
            attempts.append({"level": len(attempts), "prompt_chars": len(formatted_prompt)})
            return formatted_prompt

        prompt, level_idx = schema_service.build_prompt_with_progressive_schema_stripping(
            schema_to_use,
            encoding_model_name=llm.llm_config.model,
            max_prompt_len=max_prompt_len,
            prompt_format_func=_recording_prompt_format_func,
            item_id=data_item.question_id,
            log_prefix="SQL Generation",
        )
        # 循环在首个不超预算的 level 返回，此前的 level 均视为超预算
        for attempt in attempts:
            attempt["over_budget"] = prompt is None or attempt["level"] < level_idx
        trace_event(
            "prompt_built",
            generator=type(self).__name__,
            level=level_idx,
            prompt_available=prompt is not None,
            final_prompt_chars=0 if prompt is None else len(prompt),
            final_prompt=prompt,
            max_model_len=llm.llm_config.max_model_len,
            max_tokens=llm.llm_config.max_tokens,
            prompt_budget=max_prompt_len,
            attempts=attempts,
        )

        if prompt is not None and level_idx > 0:
            logger.warning(f"SQL Generation prompt for item {data_item.question_id} was too large. Compressed using level {level_idx}")
        return prompt, level_idx

    def _parse_llm_response(self, response: str) -> Optional[str]:
        try:
            answer_match = re.search(r"<result>(.*?)</result>", response, re.DOTALL)
            if not answer_match:
                logger.warning("No <result> tag found in LLM response")
                logger.warning(f"Response content: {response}")
                return None
            answer_content = answer_match.group(1).strip()
            # strip ```sql```
            if answer_content.startswith("```sql") and answer_content.endswith("```"):
                answer_content = answer_content[len("```sql"):-len("```")].strip()
            
            if not answer_content or not answer_content.strip():
                logger.warning("Parsed SQL content is empty")
                return None
                
            return answer_content
        except Exception as e:
            logger.error(f"Error parsing LLM response: {e}")
            return None
