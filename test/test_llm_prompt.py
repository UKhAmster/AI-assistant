"""Тесты рендеринга системного промпта."""
import pytest

from backend.engines.llm import LLMAgent


@pytest.fixture
def agent_no_retriever():
    class DummyClient:
        pass
    return LLMAgent(DummyClient(), retriever=None)


class TestSystemPrompt:
    def test_no_caller_phone_means_no_phone_block(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=1, caller_phone=None,
        )
        assert "ТЕЛЕФОН АБОНЕНТА" not in prompt

    def test_caller_phone_inserts_block(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=1, caller_phone="+79001234567",
        )
        assert "ТЕЛЕФОН АБОНЕНТА" in prompt
        assert "+79001234567" in prompt
        assert "не переспрашивай" in prompt.lower()

    def test_operator_block_present(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=1, caller_phone=None,
        )
        assert "ПЕРЕВОД НА ОПЕРАТОРА" in prompt
        assert "предложи" in prompt.lower() or "могу помочь" in prompt.lower()

    def test_year_instruction_present(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=3, caller_phone=None,
        )
        assert "год" in prompt.lower()
        assert "admission_year" in prompt or "в этом году" in prompt.lower()

    def test_discovery_stage_for_first_turns(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=1, caller_phone=None,
        )
        assert "Знакомство" in prompt or "Discovery" in prompt

    def test_inform_stage_for_later_turns(self, agent_no_retriever):
        prompt = agent_no_retriever._build_system_prompt(
            user_text="тест", user_turn_count=5, caller_phone=None,
        )
        assert "Информирование" in prompt or "Inform" in prompt
